
from pathlib import Path
from time import perf_counter
import statistics

import cv2
from ultralytics import YOLO

from quantize_onnx import MODEL_PATH


MODEL_PATH = Path(
    "runs/detect/runs/metal_nut/"
    "baseline/weights/best_256_int8.onnx"
)
IMAGE_DIR = Path("data/yolo_rebuilt/images/test")
IMAGE_SIZE = 256
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
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
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

        elapsed_ms = (perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    sorted_times = sorted(times_ms)
    p95_ms = sorted_times[int(0.95 * (len(sorted_times) - 1))]

    print("\n===== ONNX CPU BENCHMARK =====")
    print(f"Runs:              {len(times_ms)}")
    print(f"Mean latency:      {mean_ms:.2f} ms/image")
    print(f"Median latency:    {median_ms:.2f} ms/image")
    print(f"Approx. p95:       {p95_ms:.2f} ms/image")
    print(f"10 ms target met:  {'YES' if mean_ms < 10 else 'NO'}")
    print("\nTiming includes preprocessing, model execution, and postprocessing.")


if __name__ == "__main__":
    main()
