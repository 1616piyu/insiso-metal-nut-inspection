
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


MODEL_PATH = Path(
    "runs/detect/runs/metal_nut/"
    "baseline/weights/best_256.onnx"
)
OUTPUT_PATH = Path(
    "runs/detect/runs/metal_nut/"
    "baseline/weights/best_256_int8.onnx"
)
TRAIN_IMAGES = Path("data/yolo_rebuilt/images/train")
IMAGE_SIZE = 256
MAX_CALIBRATION_IMAGES = 50


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, model_path, image_paths):
        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = session.get_inputs()[0].name
        self.image_paths = image_paths
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        image = cv2.imread(str(image_path))
        if image is None:
            return self.get_next()

        # Dataset images are square, so resize directly to 256x256.
        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        # OpenCV loads BGR; the model expects RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return {self.input_name: image}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {MODEL_PATH}"
        )

    image_paths = sorted(
        path for path in TRAIN_IMAGES.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[:MAX_CALIBRATION_IMAGES]

    if not image_paths:
        raise FileNotFoundError(
            f"No training images found in {TRAIN_IMAGES}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source model: {MODEL_PATH}")
    print(f"Calibration images: {len(image_paths)}")
    print("Quantizing model to INT8...")

    reader = ImageCalibrationReader(MODEL_PATH, image_paths)

    quantize_static(
        model_input=str(MODEL_PATH),
        model_output=str(OUTPUT_PATH),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=__import__(
            "onnxruntime.quantization"
        ).quantization.CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],
    )

    print("\n===== INT8 QUANTIZATION COMPLETE =====")
    print(f"Output: {OUTPUT_PATH}")
    print(f"File exists: {OUTPUT_PATH.exists()}")

    if OUTPUT_PATH.exists():
        print(f"FP32 size: {MODEL_PATH.stat().st_size / 1e6:.2f} MB")
        print(f"INT8 size: {OUTPUT_PATH.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
