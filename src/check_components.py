
from pathlib import Path
import cv2
import numpy as np

mask_path = Path(
    "data/raw/metal_nut/ground_truth/bent/000_mask.png"
)

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(mask_path)

binary = (mask > 0).astype(np.uint8)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

print("Mask:", mask_path)
print("Connected components excluding background:", num_labels - 1)

for component_id in range(1, num_labels):
    x, y, w, h, area = stats[component_id]
    print(
        f"Component {component_id}: "
        f"x={x}, y={y}, width={w}, height={h}, area={area}"
    )

label_path = Path("data/processed/labels/defects/bent_000.txt")
print("\nCurrent YOLO label:")
print(label_path.read_text() if label_path.exists() else "Missing label")
