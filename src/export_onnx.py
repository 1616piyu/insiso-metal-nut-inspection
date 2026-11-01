
from pathlib import Path
from ultralytics import YOLO


def main():
    model_path = Path(
        "runs/detect/runs/metal_nut/"
        "baseline/weights/best.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    model = YOLO(str(model_path))

    exported_path = model.export(
        format="onnx",
        imgsz=320,
        batch=1,
        dynamic=False,
        simplify=False,
    )

    print("\n===== ONNX EXPORT COMPLETE =====")
    print(f"Exported model: {exported_path}")
    print(f"File exists: {Path(exported_path).exists()}")


if __name__ == "__main__":
    main()
