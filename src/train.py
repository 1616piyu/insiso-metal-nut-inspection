
from ultralytics import YOLO


def main():
    # Start from pretrained YOLOv8 nano weights.
    model = YOLO("yolov8n.pt")

    model.train(
        data="configs/metal_nut.yaml",
        epochs=30,
        imgsz=320,
        batch=8,
        workers=0,
        patience=8,
        seed=42,
        project="runs/metal_nut",
        name="baseline",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
