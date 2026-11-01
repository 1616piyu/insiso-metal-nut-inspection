
from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/metal_nut/improved_v2/weights/best.pt"
)

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

for conf in thresholds:
    metrics = model.val(
        data="configs/metal_nut.yaml",
        split="test",
        imgsz=320,
        batch=8,
        workers=0,
        conf=conf,
        plots=False,
        verbose=False,
    )

    print(
        f"Confidence: {conf:.2f} | "
        f"Precision: {metrics.box.mp:.4f} | "
        f"Recall: {metrics.box.mr:.4f} | "
        f"mAP50: {metrics.box.map50:.4f}"
    )
