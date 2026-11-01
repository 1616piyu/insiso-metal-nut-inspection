
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path("data/raw/metal_nut")
IMAGE_DIR = Path("data/processed/images/defects")
LABEL_DIR = Path("data/processed/labels/defects")

EXAMPLES = [
    ("bent", "000.png"),
    ("color", "000.png"),
    ("flip", "000.png"),
    ("scratch", "001.png"),
]


def main():
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for row, (category, filename) in enumerate(EXAMPLES):
        image_path = IMAGE_DIR / f"{category}_{filename}"
        mask_path = (
            SOURCE / "ground_truth" / category
            / f"{Path(filename).stem}_mask.png"
        )
        label_path = LABEL_DIR / f"{category}_{Path(filename).stem}.txt"

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(
                f"Missing image or mask for {category}/{filename}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        overlay = image.copy()
        overlay[mask > 0] = (
            0.5 * overlay[mask > 0]
            + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        with label_path.open("r", encoding="utf-8") as f:
            labels = [line.split() for line in f if line.strip()]

        for cls, xc, yc, bw, bh in labels:
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))

            x1 = round((xc - bw / 2) * width)
            y1 = round((yc - bh / 2) * height)
            x2 = round((xc + bw / 2) * width)
            y2 = round((yc + bh / 2) * height)

            cv2.rectangle(
                overlay, (x1, y1), (x2, y2), (0, 255, 0), 3
            )

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{category}: original")

        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("Ground-truth mask")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Red = mask, green = box")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
