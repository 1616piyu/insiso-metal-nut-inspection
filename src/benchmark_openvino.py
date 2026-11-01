
from pathlib import Path
from time import perf_counter
import statistics

import cv2
from ultralytics import YOLO

MODEL_PATH = Path(
    "runs/detect/runs/metal_nut/"
    "baseline/weights/best_openvino_model"
)
IMAGE_DIR = Path("data/yolo_rebuilt/images/test")
IMAGE_SIZE = 320
WARMUP_RUNS = 5
MEASURED_RUNS = 30


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images in {IMAGE_DIR}")

    image = cv2.imread(str(image_paths[0]))
    if image is None:
        raise ValueError(f"Could not read {image_paths[0]}")

    model = YOLO(str(MODEL_PATH), task="detect")

    print(f"Model: {MODEL_PATH}")
    print(f"Image: {image_paths[0].name}")
    print("Warming up...")

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=0.25,
            verbose=False,
        )

    times_ms = []
    print(f"Measuring {MEASURED_RUNS} runs...")

    for _ in range(MEASURED_RUNS):
        start = perf_counter()
        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=0.25,
            verbose=False,
        )
        times_ms.append((perf_counter() - start) * 1000)

    ordered = sorted(times_ms)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]

    print("\n===== OPENVINO CPU BENCHMARK =====")
    print(f"Runs:             {len(times_ms)}")
    print(f"Mean latency:     {statistics.mean(times_ms):.2f} ms/image")
    print(f"Median latency:   {statistics.median(times_ms):.2f} ms/image")
    print(f"Approx. p95:      {p95:.2f} ms/image")
    print(f"10 ms target met: {'YES' if statistics.mean(times_ms) < 10 else 'NO'}")
    print("Timing includes preprocessing, model execution, and postprocessing.")


if __name__ == "__main__":
    main()
