from pathlib import Path
import cv2
import random
import math

BASE = Path("data/yolo_rebuilt")
OUTPUT = Path("runs/metal_nut/label_audit")
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

for split in ["train", "val", "test"]:
    image_dir = BASE / "images" / split
    label_dir = BASE / "labels" / split

    samples = []

    for label_path in sorted(label_dir.glob("*.txt")):
        lines = [
            line.strip()
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        if lines:
            image_path = image_dir / (label_path.stem + ".png")

            if image_path.exists():
                samples.append((image_path, lines))

    random.shuffle(samples)
    samples = samples[:8]

    tiles = []

    for image_path, lines in samples:
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]

        for line in lines:
            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        image = cv2.resize(image, (350, 350))
        cv2.putText(
            image, image_path.name, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        tiles.append(image)

    if not tiles:
        print(f"No labeled images found in {split}")
        continue

    cols = 4
    rows = math.ceil(len(tiles) / cols)
    canvas = cv2.copyMakeBorder(
        cv2.vconcat([
            cv2.hconcat(
                tiles[r * cols:(r + 1) * cols]
                + [cv2.imread(str(samples[0][0]))[:350, :350]] *
                  (cols - len(tiles[r * cols:(r + 1) * cols]))
            )
            for r in range(rows)
        ]),
        10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    output_path = OUTPUT / f"{split}_defect_boxes.jpg"
    cv2.imwrite(str(output_path), canvas)
    print(f"{split}: {len(samples)} images -> {output_path}")

print("Visual label audit completed.")
