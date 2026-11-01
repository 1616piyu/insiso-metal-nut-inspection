
from pathlib import Path

import cv2
import numpy as np

SOURCE = Path("data/raw/metal_nut")
OUTPUT = Path("data/processed")

DEFECT_TYPES = ["bent", "color", "flip", "scratch"]
MIN_COMPONENT_AREA = 10


def mask_to_boxes(mask_path):
    """Convert each connected mask component into a YOLO box."""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    binary = (mask > 0).astype(np.uint8)
    image_height, image_width = binary.shape

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    boxes = []

    # Skip label 0 because it represents the background.
    for component_id in range(1, num_labels):
        x, y, box_width, box_height, area = stats[component_id]

        if area < MIN_COMPONENT_AREA:
            continue

        x_center = (x + box_width / 2) / image_width
        y_center = (y + box_height / 2) / image_height

        boxes.append(
            (
                0,
                x_center,
                y_center,
                box_width / image_width,
                box_height / image_height,
            )
        )

    return boxes


def save_labels(boxes, label_path):
    """Save YOLO labels, one bounding box per line."""

    label_path.parent.mkdir(parents=True, exist_ok=True)

    with label_path.open("w", encoding="utf-8") as file:
        for cls, xc, yc, width, height in boxes:
            file.write(
                f"{cls} {xc:.6f} {yc:.6f} "
                f"{width:.6f} {height:.6f}\n"
            )


def main():
    total_images = 0
    total_boxes = 0

    for defect_type in DEFECT_TYPES:
        image_dir = SOURCE / "test" / defect_type
        mask_dir = SOURCE / "ground_truth" / defect_type

        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Missing image or mask directory for {defect_type}"
            )

        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / f"{image_path.stem}_mask.png"

            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")

            boxes = mask_to_boxes(mask_path)

            if not boxes:
                raise ValueError(f"No valid components: {mask_path}")

            # Write labels directly. Do not copy or overwrite images.
            # The existing processed images are left untouched.
            label_path = (
                OUTPUT / "labels" / "defects"
                / f"{defect_type}_{image_path.stem}.txt"
            )

            save_labels(boxes, label_path)

            total_images += 1
            total_boxes += len(boxes)

            print(f"{defect_type}/{image_path.name}: {len(boxes)} box(es)")

    print("\nConversion completed.")
    print(f"Images annotated: {total_images}")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Labels saved in: {OUTPUT / 'labels' / 'defects'}")


if __name__ == "__main__":
    main()
