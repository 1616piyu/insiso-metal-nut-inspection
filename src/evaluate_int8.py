
from ultralytics import YOLO


def main():
    model = YOLO(
        "runs/detect/runs/metal_nut/"
        "baseline/weights/best_256_int8.onnx",
        task="detect",
    )

    results = model.val(
        data="configs/metal_nut.yaml",
        split="test",
        imgsz=256,
        batch=1,
        workers=0,
        plots=True,
        project="runs/metal_nut",
        name="int8_256_test",
        exist_ok=True,
    )

    print("\n===== INT8 TEST RESULTS =====")
    print(f"Precision:       {results.box.mp:.4f}")
    print(f"Recall:          {results.box.mr:.4f}")
    print(f"mAP@0.5:         {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95:    {results.box.map:.4f}")


if __name__ == "__main__":
    main()
