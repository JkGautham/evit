import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
from scipy.ndimage import center_of_mass

def visualize_segmentation_sample(img, seg_mask, title_prefix="Sample"):
    """
    Visualize a sample with original image, segmentation mask, and overlay.
    
    Args:
        img (Tensor): [3, H, W] RGB image tensor.
        seg_mask (Tensor): [H, W] segmentation mask tensor.
        title_prefix (str): Prefix for subplot titles.
    """
    img_np = TF.to_pil_image(img)
    seg_np = seg_mask.numpy()

    # Define colormap (up to 10 classes + background)
    colors = plt.cm.get_cmap("tab10", np.max(seg_np)+1)

    # RGBA overlay (for transparent mask on image)
    overlay = np.zeros((seg_np.shape[0], seg_np.shape[1], 4), dtype=np.float32)

    # RGB mask (for pure mask image)
    seg_rgb = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.float32)

    # Apply color to each class ID
    unique_ids = np.unique(seg_np)
    for cat_id in unique_ids:
        if cat_id == 0:
            continue  # skip background
        mask = seg_np == cat_id
        rgba = colors(cat_id)
        overlay[mask] = rgba
        seg_rgb[mask] = rgba[:3]

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axs[0].imshow(img_np)
    axs[0].set_title(f"{title_prefix} - Original Image")
    axs[0].axis("off")

    # Segmentation mask only
    axs[1].imshow(seg_rgb)
    axs[1].set_title(f"{title_prefix} - Segmentation Mask")
    axs[1].axis("off")

    # Overlay with numeric labels
    axs[2].imshow(img_np)
    axs[2].imshow(overlay, alpha=0.4)

    for cat_id in unique_ids:
        if cat_id == 0:
            continue
        mask = seg_np == cat_id
        if np.any(mask):
            y, x = center_of_mass(mask)
            axs[2].text(x, y, str(cat_id), fontsize=10, color='white', ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1.5))

    axs[2].set_title(f"{title_prefix} - Overlay + Class IDs")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
