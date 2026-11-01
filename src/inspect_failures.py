
from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = (
    "runs/detect/runs/metal_nut/"
    "baseline/weights/best.pt"
)
IMAGE_DIR = Path("data/yolo_rebuilt/images/test")
LABEL_DIR = Path("data/yolo_rebuilt/labels/test")
OUTPUT_DIR = Path("runs/metal_nut/failure_cases")

IMAGE_SIZE = 320
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.5


def load_ground_truth(label_path, width, height):
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        _, xc, yc, bw, bh = map(float, parts)
        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)
        boxes.append([x1, y1, x2, y2])

    return boxes


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    saved = 0

    for image_path in sorted(IMAGE_DIR.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        gt_boxes = load_ground_truth(label_path, width, height)
        # Skip good images; focus on images with annotated defects.
        if not gt_boxes:
            continue

        result = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            verbose=False,
        )[0]

        predictions = []
        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                predictions.append([int(v) for v in box])

        # Match each ground-truth box to its best prediction.
        matched_predictions = set()
        missed_gt = []

        for gt in gt_boxes:
            candidates = [
                (box_iou(gt, pred), idx)
                for idx, pred in enumerate(predictions)
                if idx not in matched_predictions
            ]

            best_iou, best_idx = max(candidates, default=(0.0, -1))
            if best_iou >= IOU_THRESHOLD:
                matched_predictions.add(best_idx)
            else:
                missed_gt.append(gt)

        false_positives = [
            pred for idx, pred in enumerate(predictions)
            if idx not in matched_predictions
        ]

        # Keep cases with a missed defect or false positive.
        if not missed_gt and not false_positives:
            continue

        annotated = image.copy()

        # Ground truth: green
        for x1, y1, x2, y2 in gt_boxes:
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), (0, 200, 0), 2
            )

        # Predictions: red
        for x1, y1, x2, y2 in predictions:
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), (0, 0, 255), 2
            )

        cv2.putText(
            annotated,
            f"GT: {len(gt_boxes)}  Pred: {len(predictions)}  "
            f"Missed: {len(missed_gt)}  FP: {len(false_positives)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
        )

        output_path = OUTPUT_DIR / f"failure_{saved + 1}_{image_path.name}"
        cv2.imwrite(str(output_path), annotated)

        print(
            f"{image_path.name}: GT={len(gt_boxes)}, "
            f"Pred={len(predictions)}, Missed={len(missed_gt)}, "
            f"False positives={len(false_positives)}"
        )

        saved += 1
        if saved >= 10:
            break

    print(f"\nSaved {saved} failure cases to: {OUTPUT_DIR.resolve()}")
    if saved < 5:
        print("Fewer than 5 cases met this script's failure criteria.")


if __name__ == "__main__":
    main()
