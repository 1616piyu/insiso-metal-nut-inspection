
from ultralytics import YOLO


def main():
    model = YOLO(
        "runs/detect/runs/metal_nut/improved_v2/weights/best.pt"
    )

    results = model.val(
        data="configs/metal_nut.yaml",
        split="test",
        imgsz=320,
        batch=8,
        workers=0,
        plots=True,
        project="runs/metal_nut",
        name="improved_v2_test",
        exist_ok=True,
    )

    print("\n===== IMPROVED V2 TEST RESULTS =====")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
