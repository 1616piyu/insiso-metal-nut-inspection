
from pathlib import Path
from ultralytics import YOLO

def main():
    model_path = Path(
        "runs/detect/runs/metal_nut/"
        "baseline/weights/best_256.pt"
    )

    model = YOLO(str(model_path))
    exported_path = model.export(
        format="onnx",
        imgsz=256,
        batch=1,
        dynamic=False,
        simplify=False,
    )

    print(f"Exported model: {exported_path}")
    print(f"File exists: {Path(exported_path).exists()}")

if __name__ == "__main__":
    main()
