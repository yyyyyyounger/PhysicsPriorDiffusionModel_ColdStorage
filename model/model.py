import logging
from collections import OrderedDict
#import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import model.networks as networks
import model.networkHelper as Helpernetwork
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.netH = self.set_device(Helpernetwork.MPRfusion())
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        self.finetune_netH = opt['model'].get('finetune_netH', False)
        train_opt = opt.get('train') or {}
        self.lambda_t = train_opt.get('lambda_t', 0.01)
        self.lambda_asm = train_opt.get('lambda_asm', 0.05)
        self.log_dict = OrderedDict()
        self.current_physical_losses = OrderedDict([
            ('loss_t', 0.0),
            ('loss_asm', 0.0),
            ('loss_physical_total', 0.0),
        ])

        if self.opt['phase'] == 'train':
            self.netG.train()
            self.netH.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info('Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            if self.finetune_netH:
                netH_unfreeze_prefixes = (
                    'layer0_.', 'layer1_.', 'layer2_.', 'layer3_.', 'layer4_.',
                    'layer_0_.', 'layer_1_.', 'layer_2_.', 'layer_3_.',
                    'res0_.', 'res1_.', 'res2_.', 'res3_.', 'res4_.',
                    'res_0_.', 'res_1_.', 'res_2_.', 'res_3_.',
                    'fusion1_.', 'fusion2_.', 'fusion3_.', 'fusion4_.',
                    'fusion_0_.', 'fusion_1_.', 'fusion_2_.', 'fusion_3_.',
                    'csff_', 'sam.', 'concat.', 'last.', 'conv_T_', 'ANet.',
                )
                netH_finetune_params = []
                netH_frozen_count = 0
                for k, v in self.netH.named_parameters():
                    if k.startswith(netH_unfreeze_prefixes):
                        v.requires_grad = True
                        netH_finetune_params.append(v)
                    else:
                        v.requires_grad = False
                        netH_frozen_count += 1
                logger.info('NetH: unfreezing {} params, freezing {} params'.format(
                    len(netH_finetune_params), netH_frozen_count))

                lr_netH = opt['train']['optimizer'].get('lr_netH', 1e-5)
                param_groups = [
                    {'params': optim_params, 'lr': opt['train']['optimizer']['lr']},
                    {'params': netH_finetune_params, 'lr': lr_netH},
                ]
                self.optG = torch.optim.Adam(param_groups)
            else:
                for v in self.netH.parameters():
                    v.requires_grad = False
                self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        if isinstance(data, dict):
            self.data = {}
            for key, item in data.items():
                if item is None:
                    self.data[key] = None
                elif torch.is_tensor(item):
                    self.data[key] = item.to(self.device)
                elif key in ('depth', 'beta'):
                    self.data[key] = torch.as_tensor(item).to(self.device)
                else:
                    self.data[key] = item
        else:
            self.data = self.set_device(data)

    def _has_physical_targets(self):
        return (
            hasattr(self, 'data')
            and isinstance(self.data, dict)
            and self.data.get('depth') is not None
            and self.data.get('beta') is not None
            and hasattr(self, 'out_T')
            and hasattr(self, 'out_I')
        )

    def _zero_physical_loss(self):
        if hasattr(self, 'out_T'):
            return self.out_T.new_tensor(0.0)
        return torch.tensor(0.0, device=self.device)

    def _prepare_depth_and_beta(self):
        depth = self.data['depth'].to(device=self.out_T.device, dtype=self.out_T.dtype)
        beta = self.data['beta'].to(device=self.out_T.device, dtype=self.out_T.dtype)
        batch_size = self.out_T.size(0)

        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            if depth.size(0) == batch_size:
                depth = depth.unsqueeze(1)
            else:
                depth = depth.unsqueeze(0)
        elif depth.dim() != 4:
            raise ValueError('depth must have shape [B,H,W], [B,1,H,W], or [H,W].')

        if depth.size(0) == 1 and batch_size != 1:
            depth = depth.expand(batch_size, -1, -1, -1)
        if depth.size(0) != batch_size:
            raise ValueError('depth batch size must match out_T batch size.')

        if beta.numel() == 1:
            beta = beta.reshape(1, 1, 1, 1).expand(batch_size, 1, 1, 1)
        else:
            beta = beta.reshape(batch_size, 1, 1, 1)

        if depth.shape[-2:] != self.out_T.shape[-2:]:
            depth = F.interpolate(
                depth, size=self.out_T.shape[-2:], mode='bilinear', align_corners=False)
        return depth, beta

    def _compute_physical_losses(self, hazy_input_01=None):
        if not self._has_physical_targets():
            zero_loss = self._zero_physical_loss()
            return zero_loss, zero_loss

        depth, beta = self._prepare_depth_and_beta()
        t_gt = torch.exp(-beta * depth)
        out_T = torch.clamp(self.out_T, min=1e-4, max=1.0)
        t_gt = torch.clamp(t_gt, min=1e-4, max=1.0)
        loss_t = F.l1_loss(out_T, t_gt)

        if hazy_input_01 is None:
            hazy_input_01 = (self.data['SR'] + 1.0) / 2.0
        hazy_input_01 = hazy_input_01.to(device=self.out_I.device, dtype=self.out_I.dtype)
        if hazy_input_01.shape[-2:] != self.out_I.shape[-2:]:
            hazy_input_01 = F.interpolate(
                hazy_input_01, size=self.out_I.shape[-2:], mode='bilinear', align_corners=False)
        loss_asm = F.l1_loss(self.out_I, hazy_input_01)
        return loss_t, loss_asm

    @torch.no_grad()
    def compute_current_physical_losses(self):
        loss_t, loss_asm = self._compute_physical_losses()
        loss_physical_total = self.lambda_t * loss_t + self.lambda_asm * loss_asm
        return OrderedDict([
            ('loss_t', loss_t.item()),
            ('loss_asm', loss_asm.item()),
            ('loss_physical_total', loss_physical_total.item()),
        ])

    @torch.no_grad()
    def update_physical_log(self):
        self.current_physical_losses = self.compute_current_physical_losses()
        for key, value in self.current_physical_losses.items():
            self.log_dict[key] = value
        return self.current_physical_losses

    def optimize_parameters(self):
        self.optG.zero_grad()
        hazy_input_01 = (self.data['SR'] + 1.0) / 2.0
        self.output, self.stage1_output, self.out_T, self.out_A, self.out_I = self.netH(hazy_input_01)
        condition = torch.cat([self.output / 0.5 - 1, self.out_T / 0.5 - 1], dim=1)
        l_pix = self.netG(self.data['HR'], condition)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)

        loss_t = l_pix.new_tensor(0.0)
        loss_asm = l_pix.new_tensor(0.0)
        if self.finetune_netH and self._has_physical_targets():
            loss_t, loss_asm = self._compute_physical_losses(hazy_input_01=hazy_input_01)
        loss_physical_total = self.lambda_t * loss_t + self.lambda_asm * loss_asm
        total_loss = l_pix + loss_physical_total

        total_loss.backward()
        self.optG.step()
        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['loss_t'] = loss_t.item()
        self.log_dict['loss_asm'] = loss_asm.item()
        self.log_dict['loss_physical_total'] = loss_physical_total.item()
        self.log_dict['loss_total'] = total_loss.item()

    def test(self, continous=False):
        self.netG.eval()
        self.netH.eval()
        with torch.no_grad():
            self.output, self.stage1_output, self.out_T, self.out_A, self.out_I = self.netH((self.data['SR'] + 1.0) / 2.0)
            condition = torch.cat([self.output/0.5 - 1, self.out_T/0.5 - 1], dim=1)
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(condition)
            else:
                self.SR = self.netG.super_resolution(condition)
            self.update_physical_log()

        self.netG.train()
        self.netH.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['Out'] = torch.clamp((self.SR + 1.0) / 2.0, min=0.0, max=1.0).detach().float().cpu()
            out_dict['LR'] = torch.clamp((self.data['SR'] + 1.0) / 2.0, min=0.0, max=1.0).detach().float().cpu()
            out_dict['HR'] = torch.clamp((self.data['HR'] + 1.0) / 2.0, min=0.0, max=1.0).detach().float().cpu()
            if hasattr(self, 'out_T'):
                out_dict['out_T'] = torch.clamp(self.out_T, min=0.0, max=1.0).detach().float().cpu()
            if hasattr(self, 'out_A'):
                out_dict['out_A'] = torch.clamp(self.out_A, min=0.0, max=1.0).detach().float().cpu()
            if hasattr(self, 'out_I'):
                out_dict['out_I'] = torch.clamp(self.out_I, min=0.0, max=1.0).detach().float().cpu()
            if hasattr(self, 'stage1_output'):
                out_dict['stage1_output'] = torch.clamp(self.stage1_output, min=0.0, max=1.0).detach().float().cpu()
            if hasattr(self, 'output'):
                out_dict['output'] = torch.clamp(self.output, min=0.0, max=1.0).detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        sH, nH = self.get_network_description(self.netH)
        if isinstance(self.netH, nn.DataParallel):
            net_struc_strH = '{} - {}'.format(self.netH.__class__.__name__, self.netH.module.__class__.__name__)
        else:
            net_struc_strH = '{}'.format(self.netH.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info('Network H structure: {}, with parameters: {:,d}'.format(net_struc_strH, nH))
        #logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info('Saved model in [{:s}] ...'.format(gen_path))

        if self.finetune_netH:
            netH_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{}_netH.pth'.format(iter_step, epoch))
            state_dictH = self.netH.state_dict()
            for key, param in state_dictH.items():
                state_dictH[key] = param.cpu()
            torch.save(state_dictH, netH_path)
            logger.info('Saved netH in [{:s}] ...'.format(netH_path))

    def load_network(self):
        load_pathG = self.opt['path']['resume_state']
        if load_pathG is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_pathG))
            gen_path = '{}_gen.pth'.format(load_pathG)
            opt_path = '{}_opt.pth'.format(load_pathG)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            net = torch.load(gen_path, map_location=self.device)
            if self.opt['phase'] == 'train':
                model_dict = network.state_dict()
                pretrained_dict = {
                    k: v
                    for k, v in net.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                skipped = set(net.keys()) - set(pretrained_dict.keys())
                if skipped:
                    sample = list(skipped)[:12]
                    logger.info(
                        'Skipped {} pretrained G keys (missing or shape mismatch). Sample: {}'.format(
                            len(skipped), sample))
                model_dict.update(pretrained_dict)
                network.load_state_dict(model_dict)
            else:
                model_dict = network.state_dict()
                schedule_buffer_keys = {
                    'betas',
                    'alphas_cumprod',
                    'alphas_cumprod_prev',
                    'sqrt_alphas_cumprod',
                    'sqrt_one_minus_alphas_cumprod',
                    'log_one_minus_alphas_cumprod',
                    'sqrt_recip_alphas_cumprod',
                    'sqrt_recipm1_alphas_cumprod',
                    'posterior_variance',
                    'posterior_log_variance_clipped',
                    'posterior_mean_coef1',
                    'posterior_mean_coef2',
                }
                skipped_schedule = [
                    k for k, v in net.items()
                    if k in schedule_buffer_keys
                    and k in model_dict
                    and v.shape != model_dict[k].shape
                ]
                if skipped_schedule:
                    load_dict = {
                        k: v for k, v in net.items()
                        if k not in skipped_schedule
                    }
                    incompatible = network.load_state_dict(load_dict, strict=False)
                    unexpected = list(incompatible.unexpected_keys)
                    missing = [
                        k for k in incompatible.missing_keys
                        if k not in skipped_schedule
                    ]
                    if unexpected or missing:
                        raise RuntimeError(
                            'Error(s) in loading state_dict for {}:\n'
                            '\tMissing keys: {}\n\tUnexpected keys: {}'.format(
                                network.__class__.__name__, missing, unexpected))
                    logger.info(
                        'Skipped {} schedule buffers with mismatched shapes: {}'.format(
                            len(skipped_schedule), skipped_schedule))
                else:
                    network.load_state_dict(net)

            if self.opt['phase'] == 'train' and os.path.isfile(opt_path):
                ckpt = torch.load(opt_path, map_location=self.device)
                if ckpt.get('optimizer') is not None:
                    self.optG.load_state_dict(ckpt['optimizer'])
                self.begin_step = ckpt.get('iter', self.begin_step)
                self.begin_epoch = ckpt.get('epoch', self.begin_epoch)
                logger.info(
                    'Loaded training state from [{:s}] (epoch={}, iter={}).'.format(
                        opt_path, self.begin_epoch, self.begin_step))
            elif self.opt['phase'] == 'train':
                logger.info(
                    'No optimizer state at [{:s}]; starting from iter=0, epoch=0 (weights only).'.format(
                        opt_path))

        load_pathH_ft = self.opt['path'].get('resume_stateH_finetune')
        load_pathH = self.opt['path']['resume_stateH']

        if load_pathH_ft is not None:
            logger.info('Loading finetuned netH from [{:s}] ...'.format(load_pathH_ft))
            network = self.netH
            state_dict = torch.load(load_pathH_ft, map_location=lambda storage, loc: storage)
            network.load_state_dict(state_dict)
        elif load_pathH is not None:
            logger.info('Loading pretrained model for H [{:s}] ...'.format(load_pathH))
            network = self.netH
            load_net = torch.load(load_pathH, map_location=lambda storage, loc: storage)
            load_net = load_net['params']
            for k, v in deepcopy(load_net).items():
                if k.startswith('module.'):
                    load_net[k[7:]] = v
                    load_net.pop(k)
            network.load_state_dict(load_net, strict=(not self.opt['model']['finetune_norm']))




# class DDPM(BaseModel):
#     def __init__(self, opt):
#         super(DDPM, self).__init__(opt)
#         # define network and load pretrained models
#         self.netG = self.set_device(networks.define_G(opt))
#         self.schedule_phase = None
#
#         # set loss and load resume state
#         self.set_loss()
#         self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
#         if self.opt['phase'] == 'train':
#             self.netG.train()
#             # find the parameters to optimize
#             if opt['model']['finetune_norm']:
#                 optim_params = []
#                 for k, v in self.netG.named_parameters():
#                     v.requires_grad = False
#                     if k.find('transformer') >= 0:
#                         v.requires_grad = True
#                         v.data.zero_()
#                         optim_params.append(v)
#                         logger.info('Params [{:s}] initialized to 0 and will optimize.'.format(k))
#             else:
#                 optim_params = list(self.netG.parameters())
#
#             self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
#             self.log_dict = OrderedDict()
#         self.load_network()
#         self.print_network()
#
#     def feed_data(self, data):
#         self.data = self.set_device(data)
#
#     def optimize_parameters(self):
#         self.optG.zero_grad()
#         l_pix = self.netG(self.data)
#         # need to average in multi-gpu
#         b, c, h, w = self.data['HR'].shape
#         l_pix = l_pix.sum()/int(b*c*h*w)
#         l_pix.backward()
#         self.optG.step()
#         # set log
#         self.log_dict['l_pix'] = l_pix.item()
#
#     def test(self, continous=False):
#         self.netG.eval()
#         with torch.no_grad():
#             if isinstance(self.netG, nn.DataParallel):
#                 self.SR = self.netG.module.super_resolution(self.data, continous)
#             else:
#                 self.SR = self.netG.super_resolution(self.data, continous)
#         self.netG.train()
#
#     def sample(self, batch_size=1, continous=False):
#         self.netG.eval()
#         with torch.no_grad():
#             if isinstance(self.netG, nn.DataParallel):
#                 self.SR = self.netG.module.sample(batch_size, continous)
#             else:
#                 self.SR = self.netG.sample(batch_size, continous)
#         self.netG.train()
#
#     def set_loss(self):
#         if isinstance(self.netG, nn.DataParallel):
#             self.netG.module.set_loss(self.device)
#         else:
#             self.netG.set_loss(self.device)
#
#     def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
#         if self.schedule_phase is None or self.schedule_phase != schedule_phase:
#             self.schedule_phase = schedule_phase
#             if isinstance(self.netG, nn.DataParallel):
#                 self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
#             else:
#                 self.netG.set_new_noise_schedule(schedule_opt, self.device)
#
#     def get_current_log(self):
#         return self.log_dict
#
#     def get_current_visuals(self, need_LR=True, sample=False):
#         out_dict = OrderedDict()
#         if sample:
#             out_dict['SAM'] = self.SR.detach().float().cpu()
#         else:
#             out_dict['Out'] = self.SR.detach().float().cpu()
#             out_dict['LR'] = self.data['SR'].detach().float().cpu()
#             out_dict['HR'] = self.data['HR'].detach().float().cpu()
#         return out_dict
#
#     def print_network(self):
#         s, n = self.get_network_description(self.netG)
#         if isinstance(self.netG, nn.DataParallel):
#             net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
#         else:
#             net_struc_str = '{}'.format(self.netG.__class__.__name__)
#
#         logger.info(
#             'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
#         logger.info(s)
#
#     def save_network(self, epoch, iter_step):
#         gen_path = os.path.join(
#             self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
#         opt_path = os.path.join(
#             self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
#         # gen
#         network = self.netG
#         if isinstance(self.netG, nn.DataParallel):
#             network = network.module
#         state_dict = network.state_dict()
#         for key, param in state_dict.items():
#             state_dict[key] = param.cpu()
#         torch.save(state_dict, gen_path)
#         # opt
#         opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
#         opt_state['optimizer'] = self.optG.state_dict()
#         torch.save(opt_state, opt_path)
#
#         logger.info('Saved model in [{:s}] ...'.format(gen_path))
#
#     def load_network(self):
#         load_path = self.opt['path']['resume_state']
#         if load_path is not None:
#             logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path))
#             gen_path = '{}_gen.pth'.format(load_path)
#             opt_path = '{}_opt.pth'.format(load_path)
#             # gen
#             network = self.netG
#             if isinstance(self.netG, nn.DataParallel):
#                 network = network.module
#             network.load_state_dict(torch.load(gen_path), strict=(not self.opt['model']['finetune_norm']))
#             # network.load_state_dict(torch.load(
#             #     gen_path), strict=False)
#             if self.opt['phase'] == 'train':
#                 # optimizer
#                 opt = torch.load(opt_path)
#                 self.optG.load_state_dict(opt['optimizer'])
#                 self.begin_step = opt['iter']
#                 self.begin_epoch = opt['epoch']
