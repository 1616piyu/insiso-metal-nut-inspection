
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATASET = Path("data/raw/metal_nut")


def visualize_defect(defect_type):
    image_dir = DATASET / "test" / defect_type
    mask_dir = DATASET / "ground_truth" / defect_type

    # Select the first image in this defect category.
    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        print(f"No images found for {defect_type}")
        return

    image_path = image_paths[0]

    # MVTec masks normally use the original filename
    # with '_mask' added before the extension.
    mask_path = mask_dir / f"{image_path.stem}_mask.png"

    if not mask_path.exists():
        print(f"Mask not found: {mask_path}")
        return

    # OpenCV reads color images in BGR format.
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Could not read mask: {mask_path}")
        return

    # Convert mask pixels to a simple binary representation.
    binary_mask = mask > 0

    # Overlay the mask on the original image.
    overlay = image.copy()
    overlay[binary_mask] = (
        0.5 * overlay[binary_mask]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Original: {defect_type}")

    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title("Ground-truth mask")

    axes[2].imshow(overlay)
    axes[2].set_title("Defect overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("Image:", image_path.name)
    print("Image dimensions:", image.shape)
    print("Mask dimensions:", mask.shape)
    print("Unique mask pixel values:", np.unique(mask))
    print("Defect pixels:", np.count_nonzero(binary_mask))


if __name__ == "__main__":
    for category in ["bent", "color", "flip", "scratch"]:
        visualize_defect(category)
