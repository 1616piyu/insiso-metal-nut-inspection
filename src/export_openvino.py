
from pathlib import Path
from ultralytics import YOLO


def main():
    model_path = Path(
        "runs/detect/runs/metal_nut/"
        "baseline/weights/best.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = YOLO(str(model_path))

    exported_path = model.export(
        format="openvino",
        imgsz=320,
        half=False,
        int8=False,
    )

    print("\n===== OPENVINO EXPORT COMPLETE =====")
    print(f"Exported model: {exported_path}")
    print(f"Directory exists: {Path(exported_path).exists()}")


if __name__ == "__main__":
    main()
