# Discrete-time DPM-Solver++ (multistep, order 2) for VP diffusion with noise prediction.
# Adapted from https://github.com/LuChengTHU/dpm-solver (Apache-2.0).

import torch


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


def interpolate_fn(x, xp, yp):
    """Piecewise linear interpolation; x: [N,C], xp/yp: [C,K]."""
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    return start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)


class NoiseScheduleVP:
    """Discrete VP schedule: q(x_n|x_0) with sqrt(α̅_n) as marginal alpha in solver notation."""

    def __init__(self, alphas_cumprod, eps=1e-20):
        ap = alphas_cumprod.flatten().float()
        log_alphas = 0.5 * torch.log(ap.clamp(min=eps))
        self.T = 1.0
        self.log_alpha_array = log_alphas.reshape(1, -1)
        self.total_N = self.log_alpha_array.shape[1]
        self.t_array = torch.linspace(
            0.0, 1.0, self.total_N + 1, dtype=torch.float32, device=ap.device
        )[1:].reshape(1, -1)

    def marginal_log_mean_coeff(self, t):
        return interpolate_fn(
            t.reshape((-1, 1)),
            self.t_array.to(t.device),
            self.log_alpha_array.to(t.device),
        ).reshape((-1,))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)).clamp(max=1.0 - 1e-6))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff).clamp(max=1.0 - 1e-6))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        log_alpha = -0.5 * torch.logaddexp(
            torch.zeros((1,), device=lamb.device, dtype=lamb.dtype), -2.0 * lamb
        )
        t = interpolate_fn(
            log_alpha.reshape((-1, 1)),
            torch.flip(self.log_alpha_array.to(lamb.device), [1]),
            torch.flip(self.t_array.to(lamb.device), [1]),
        )
        return t.reshape((-1,))


class DPM_Solver_PlusPlus:
    """DPM-Solver++ with algorithm_type fixed; multistep sampling order 2 only."""

    def __init__(self, noise_pred_fn, noise_schedule, clip_denoised=True):
        self.noise_pred_fn = noise_pred_fn
        self.noise_schedule = noise_schedule
        self.clip_denoised = clip_denoised

    def data_prediction_fn(self, x, t):
        noise = self.noise_pred_fn(x, t)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)
        x0 = (x - expand_dims(sigma_t, x.dim()) * noise) / expand_dims(alpha_t, x.dim())
        if self.clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def model_fn(self, x, t):
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            tb = t.expand(x.shape[0])
        else:
            tb = t
        return self.data_prediction_fn(x, tb)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1, device=device)
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T, device=device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0, device=device))
            logsnr = torch.linspace(lambda_T, lambda_0, N + 1, device=device)
            return self.noise_schedule.inverse_lambda(logsnr)
        raise ValueError("skip_type must be 'time_uniform' or 'logSNR'")

    def dpm_solver_first_update(self, x, s, t, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = self.model_fn(x, s)
        x_out = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
        return x_out, model_s

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1 = ns.marginal_lambda(t_prev_1)
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        d1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h)
        if solver_type == "dpmsolver":
            x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * d1_0
        elif solver_type == "taylor":
            x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 + (alpha_t * (phi_1 / h + 1.0)) * d1_0
        else:
            raise ValueError(solver_type)
        return x_t

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type="dpmsolver"):
        if order == 1:
            x_t, _ = self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
            return x_t
        if order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        raise ValueError("order must be 1 or 2")

    def sample_multistep(
        self,
        x,
        steps,
        skip_type="time_uniform",
        order=2,
        solver_type="dpmsolver",
        lower_order_final=True,
        denoise_to_zero=False,
    ):
        device = x.device
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
        assert timesteps.shape[0] - 1 == steps
        step = 0
        t = timesteps[step]
        t_prev_list = [t]
        t_b = t.expand(x.shape[0])
        model_prev_list = [self.model_fn(x, t_b)]
        for step in range(1, order):
            t = timesteps[step]
            t_b = t.expand(x.shape[0])
            x = self.multistep_dpm_solver_update(
                x, model_prev_list, t_prev_list, t, step, solver_type=solver_type
            )
            t_prev_list.append(t)
            model_prev_list.append(self.model_fn(x, t_b))
        for step in range(order, steps + 1):
            t = timesteps[step]
            t_b = t.expand(x.shape[0])
            if lower_order_final and steps < 10:
                step_order = min(order, steps + 1 - step)
            else:
                step_order = order
            x = self.multistep_dpm_solver_update(
                x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type
            )
            for i in range(order - 1):
                t_prev_list[i] = t_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]
            t_prev_list[-1] = t
            if step < steps:
                model_prev_list[-1] = self.model_fn(x, t_b)
        if denoise_to_zero:
            t_z = torch.ones((1,), device=device, dtype=torch.float32) * t_0
            x = self.data_prediction_fn(x, t_z.expand(x.shape[0]))
        return x
