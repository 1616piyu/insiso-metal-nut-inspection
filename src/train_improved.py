
from ultralytics import YOLO


def main():

    # Start with pretrained YOLOv8 nano weights.
    model = YOLO("yolov8n.pt")

    model.train(
        # Dataset
        data="configs/metal_nut.yaml",

        # Training
        epochs=60,
        imgsz=320,
        batch=8,
        workers=0,
        patience=15,
        seed=42,

        # Optimizer and learning rate
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2,

        # Conservative augmentation for industrial defects
        hsv_h=0.005,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=5,
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        flipud=0.0,

        mosaic=0.2,
        mixup=0.0,
        erasing=0.0,
        auto_augment=None,
        close_mosaic=5,

        # Save separately from baseline
        project="runs/metal_nut",
        name="improved_v2",
        exist_ok=True,

        plots=True,
        save=True,
        val=True,
    )


if __name__ == "__main__":
    main()
