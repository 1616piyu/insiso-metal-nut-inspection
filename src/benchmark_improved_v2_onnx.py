
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_PATH = Path(
    "runs/detect/runs/metal_nut/improved_v2/weights/best.onnx"
)

IMAGE_SIZE = 256
WARMUP_RUNS = 5
BENCHMARK_RUNS = 30

# CPU execution provider
session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# YOLO input: batch, channels, height, width
dummy_input = np.random.rand(
    1, 3, IMAGE_SIZE, IMAGE_SIZE
).astype(np.float32)

print("Model:", MODEL_PATH)
print("Input size:", IMAGE_SIZE, "x", IMAGE_SIZE)
print("Provider:", session.get_providers())

# Warm-up
for _ in range(WARMUP_RUNS):
    session.run(None, {input_name: dummy_input})

# Benchmark
times = []

for _ in range(BENCHMARK_RUNS):
    start = time.perf_counter()
    session.run(None, {input_name: dummy_input})
    end = time.perf_counter()

    times.append((end - start) * 1000)

print("\n========== ONNX CPU BENCHMARK ==========")
print(f"Mean latency:   {np.mean(times):.2f} ms")
print(f"Median latency: {np.median(times):.2f} ms")
print(f"P95 latency:    {np.percentile(times, 95):.2f} ms")
print(f"Minimum:        {np.min(times):.2f} ms")
print(f"Maximum:        {np.max(times):.2f} ms")
print("========================================")
