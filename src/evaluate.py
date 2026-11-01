
from ultralytics import YOLO

def main():
    model_path = (
        "runs/detect/runs/metal_nut/"
        "baseline/weights/best.pt"
    )

    model = YOLO(model_path)

    results = model.val(
        data="configs/metal_nut.yaml",
        split="test",
        imgsz=320,
        batch=8,
        workers=0,
        plots=True,
        project="runs/metal_nut",
        name="test_evaluation",
        exist_ok=True,
    )

    print("\n===== HELD-OUT TEST RESULTS =====")
    print(f"Precision:       {results.box.mp:.4f}")
    print(f"Recall:          {results.box.mr:.4f}")
    print(f"mAP@0.5:         {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95:    {results.box.map:.4f}")

if __name__ == "__main__":
    main()
