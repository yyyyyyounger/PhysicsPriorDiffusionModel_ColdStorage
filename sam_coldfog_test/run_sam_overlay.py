import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_anns(anns, ax):
    if len(anns) == 0:
        return
    anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    h, w = anns[0]["segmentation"].shape
    overlay = np.ones((h, w, 4))
    overlay[:, :, 3] = 0

    rng = np.random.default_rng(42)
    for ann in anns:
        m = ann["segmentation"]
        color = np.concatenate([rng.random(3), [0.45]])
        overlay[m] = color

    ax.imshow(overlay)


def main():
    image_id = "CHANGE_THIS.png"  # 換成你的某個檔名

    dirs = {
        "Clear GT": "sam_eval/clear",
        "Light fog": "sam_eval/light",
        "Medium fog": "sam_eval/medium",
    }

    checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for col, (title, folder) in enumerate(dirs.items()):
        path = os.path.join(folder, image_id)
        img = read_rgb(path)
        masks = mask_generator.generate(img)

        axes[0, col].imshow(img)
        axes[0, col].set_title(title)
        axes[0, col].axis("off")

        axes[1, col].imshow(img)
        show_anns(masks, axes[1, col])
        axes[1, col].set_title(f"SAM masks: {len(masks)}")
        axes[1, col].axis("off")

    os.makedirs("sam_outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("sam_outputs/sam_overlay_compare.png", dpi=300)
    print("saved to sam_outputs/sam_overlay_compare.png")


if __name__ == "__main__":
    main()