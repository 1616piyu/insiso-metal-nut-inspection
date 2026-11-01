
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

IMAGE_DIR = Path("data/processed/images/defects")
LABEL_DIR = Path("data/processed/labels/defects")
OUTPUT_DIR = Path("results/visualizations")


def draw_box(image_path, label_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    height, width = image.shape[:2]

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    with label_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        values = line.strip().split()

        if not values:
            continue

        cls, xc, yc, bw, bh = map(float, values)

        # Convert normalized YOLO coordinates to pixels.
        x_center = xc * width
        y_center = yc * height
        box_width = bw * width
        box_height = bh * height

        x1 = round(x_center - box_width / 2)
        y1 = round(y_center - box_height / 2)
        x2 = round(x_center + box_width / 2)
        y2 = round(y_center + box_height / 2)

        # Draw the bounding box.
        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            (0, 255, 0), 3
        )

        cv2.putText(
            image,
            "defect",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(IMAGE_DIR.glob("*.png"))

    if not images:
        raise FileNotFoundError(
            f"No processed images found in {IMAGE_DIR}"
        )

    # Inspect examples from different defect categories.
    selected = [
        p for p in images
        if p.name.startswith(("bent_", "color_", "flip_", "scratch_"))
    ]

    for image_path in selected[:8]:
        label_path = LABEL_DIR / (
            image_path.stem + ".txt"
        )

        image = draw_box(image_path, label_path)

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

        plt.figure(figsize=(7, 7))
        plt.imshow(image)
        plt.title(image_path.name)
        plt.axis("off")
        plt.show()

        print("Saved:", output_path)


if __name__ == "__main__":
    main()
