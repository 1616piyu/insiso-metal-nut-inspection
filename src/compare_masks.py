from pathlib import Path
import cv2
import numpy as np

BASE = Path("data/raw/metal_nut")
OUTPUT = Path("runs/metal_nut/original_mask_audit")
OUTPUT.mkdir(parents=True, exist_ok=True)

samples = [
    ("color", "003"),
    ("scratch", "010"),
]

for defect_type, index in samples:
    image_path = BASE / "test" / defect_type / f"{index}.png"
    mask_path = BASE / "ground_truth" / defect_type / f"{index}_mask.png"

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Missing file: {image_path} or {mask_path}")
        continue

    mask_binary = mask > 0

    overlay = image.copy()
    overlay[mask_binary] = (
        0.5 * overlay[mask_binary] +
        0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([
        image,
        mask_display,
        overlay
    ])

    cv2.putText(combined, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined, "Ground Truth Mask", (710, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined, "Overlay", (1410, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    output_path = OUTPUT / f"{defect_type}_{index}_comparison.png"
    cv2.imwrite(str(output_path), combined)

    print(f"Created: {output_path}")

print("Mask comparison completed.")
