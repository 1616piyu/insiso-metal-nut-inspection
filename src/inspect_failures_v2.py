
from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL = "runs/detect/runs/metal_nut/improved_v2/weights/best.pt"
IMAGE_DIR = Path("data/yolo_rebuilt/images/test")
LABEL_DIR = Path("data/yolo_rebuilt/labels/test")
OUTPUT_DIR = Path("runs/metal_nut/improved_v2_failures")

CONF = 0.25
IOU_THRESHOLD = 0.5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL)

def load_ground_truth(label_path, width, height):
    boxes = []

    for line in label_path.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        _, xc, yc, bw, bh = map(float, parts)

        boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ])

    return boxes


def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


saved = 0

for image_path in sorted(IMAGE_DIR.glob("*.png")):
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"
    gt_boxes = load_ground_truth(label_path, width, height)

    result = model.predict(
        image,
        imgsz=320,
        conf=CONF,
        verbose=False
    )[0]

    predictions = []

    for box in result.boxes:
        predictions.append(box.xyxy[0].cpu().numpy().tolist())

    matched_gt = set()
    matched_pred = set()

    for pred_idx, pred in enumerate(predictions):
        best_iou = 0
        best_gt = None

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = calculate_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_gt is not None and best_iou >= IOU_THRESHOLD:
            matched_gt.add(best_gt)
            matched_pred.add(pred_idx)

    missed = len(gt_boxes) - len(matched_gt)
    false_positives = len(predictions) - len(matched_pred)

    if missed > 0 or false_positives > 0:
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for idx, box in enumerate(predictions):
            x1, y1, x2, y2 = map(int, box)
            color = (255, 0, 0) if idx not in matched_pred else (0, 255, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"GT:{len(gt_boxes)} Pred:{len(predictions)} Miss:{missed} FP:{false_positives}"

        cv2.putText(
            image, text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )

        saved += 1
        output_path = OUTPUT_DIR / f"failure_{saved}_{image_path.name}"
        cv2.imwrite(str(output_path), image)

        print(
            f"{image_path.name}: GT={len(gt_boxes)}, "
            f"Pred={len(predictions)}, Missed={missed}, "
            f"False positives={false_positives}"
        )

print(f"\nTotal failure cases saved: {saved}")
print(f"Output folder: {OUTPUT_DIR}")
