# Metal Nut Defect Detection Using YOLOv8

## Computer Vision Internship Assignment – InSiSo Technologies

**Candidate:** Piyush Pratap Singh
**Project:** Metal Nut Defect Detection
**Dataset:** MVTec Anomaly Detection – Metal Nut
**Framework:** Ultralytics YOLOv8
**Language:** Python
**Hardware:** 12th Gen Intel Core i5-1235U CPU
**Repository:** [Add your public GitHub repository URL here]

---

## 1. Project Overview

The objective of this project was to build a computer vision system that can identify defective regions on metal nuts.

I used the metal nut category from the MVTec Anomaly Detection (MVTec AD) dataset. The dataset contains normal metal nut images and defective examples, including bent, color, flip, and scratch defects.

I approached the task as an object detection problem. Since the original dataset provides pixel-level defect masks rather than bounding-box annotations, I first converted those masks into YOLO-format bounding boxes. I then trained a lightweight YOLOv8n model, evaluated its performance, inspected its failure cases, and exported the trained model to ONNX for CPU inference testing.

I also experimented with training settings and input resolution to understand the trade-off between detection accuracy and inference speed.

**Dataset source:** [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## 2. Dataset Understanding and Preparation

### Dataset structure

The original `metal_nut` dataset contains normal images and four defect categories.

| Dataset category | Number of images |
|---|---:|
| Train – Good | 220 |
| Test – Good | 22 |
| Test – Bent | 25 |
| Test – Color | 22 |
| Test – Flip | 23 |
| Test – Scratch | 23 |
| **Total** | **335** |

There are 93 defective images and 93 corresponding pixel-level masks.

All original images are 700 × 700 pixels.

### Converting segmentation masks into bounding boxes

The original dataset provides segmentation masks that indicate defective pixels. However, YOLO object detection requires bounding-box annotations.

I wrote a Python script, `src/convert_masks.py`, to convert the masks into YOLO-format labels.

**Steps followed:**

1. Read each defect mask in grayscale.
2. Convert the mask into a binary image using a threshold greater than zero.
3. Apply connected component analysis to identify separate defect regions.
4. Calculate the bounding box for each connected component.
5. Ignore components smaller than 10 pixels in area.
6. Normalize the bounding-box coordinates using the image width and height.
7. Save the annotations in YOLO text format.

I used OpenCV's `connectedComponentsWithStats()` with 8-connectivity.

### Why I selected connected component analysis

A defect mask can contain multiple disconnected regions. Instead of creating one large bounding box around the entire mask, connected component analysis allows separate defect regions to receive individual bounding boxes.

This helps preserve the spatial location of separate defects and avoids unnecessarily large boxes when defects are disconnected.

Very small components were filtered out using a minimum area of 10 pixels to reduce noise.

### YOLO annotation format

Each annotation follows this format:

```text
class_id x_center y_center width height
```

All coordinates are normalized between 0 and 1.

I used a single class:

```text
0 = defect
```

This means the model detects defective regions but does not distinguish between bent, color, flip, and scratch defect types.

### Conversion results

| Metric | Result |
|---|---:|
| Defective images processed | 93 |
| Bounding boxes generated | 132 |
| Invalid annotations found | 0 |
| Minimum component area | 10 pixels |
| Number of detection classes | 1 |

Normal images have empty label files because they do not contain annotated defects.

### Train, validation, and test split

I prepared a custom YOLO dataset split:

| Split | Images | Images with defects | Images without defects |
|---|---:|---:|---:|
| Train | 263 | 65 | 198 |
| Validation | 36 | 14 | 22 |
| Test | 36 | 14 | 22 |
| **Total** | **335** | **93** | **242** |

The same split was used for the baseline and improved model experiments.

**Important limitation:** This is a custom split, not an untouched official MVTec test benchmark. The custom training split includes some images originating from the original MVTec test categories. Therefore, the reported results should be treated as preliminary results on this custom split rather than as official benchmark performance.

---

## 3. Model Selection and Training

### Why YOLOv8n?

I selected YOLOv8n because it is a lightweight object detection model.

My main considerations were:

- It is suitable for experimenting on a CPU-based laptop.
- It has a relatively small model size.
- It supports ONNX export.
- It provides bounding-box predictions that can be evaluated using precision, recall, and mAP.

I started with the pretrained `yolov8n.pt` model and fine-tuned it on the prepared metal nut dataset.

### Experiment 1: Baseline model

For the baseline, I used the following configuration:

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Initial weights | yolov8n.pt |
| Epochs | 30 |
| Image size | 320 × 320 |
| Batch size | 8 |
| Optimizer | Default YOLO optimizer |
| Workers | 0 |
| Early stopping patience | 8 |
| Random seed | 42 |

The baseline training stopped early. The best checkpoint was saved during the first epoch according to the training fitness score.

### Experiment 2: Improved training configuration

After evaluating the baseline, I experimented with a different optimizer, learning rate, and data augmentation settings.

The objective was to improve the model's ability to detect small and varied defect regions.

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Initial weights | yolov8n.pt |
| Maximum epochs | 60 |
| Image size | 320 × 320 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Final learning-rate factor | 0.01 |
| Weight decay | 0.0005 |
| Warmup epochs | 2 |
| Early stopping patience | 15 |
| Random seed | 42 |

I also used the following augmentation settings:

- Horizontal flip: 0.5
- Vertical flip: 0.0
- Rotation: 5 degrees
- Translation: 0.05
- Scale: 0.2
- Mosaic: 0.2
- MixUp: 0.0
- HSV hue: 0.005
- HSV saturation: 0.2
- HSV value: 0.2

The improved training run stopped early after 41 epochs. The best checkpoint was selected based on the training validation fitness score.

---

## 4. Model Evaluation

I evaluated both models on the same custom test split.

The main evaluation metrics were:

- **Precision:** How many predicted defect detections were correct.
- **Recall:** How many actual defect regions were detected.
- **mAP@0.5:** Mean average precision at an IoU threshold of 0.5.
- **mAP@0.5:0.95:** Mean average precision averaged across IoU thresholds from 0.5 to 0.95.

### Test results

| Metric | Baseline YOLOv8n | Improved YOLOv8n |
|---|---:|---:|
| Precision | 0.9490 | 0.5914 |
| Recall | 0.1600 | 0.5200 |
| mAP@0.5 | 0.1629 | 0.4698 |
| mAP@0.5:0.95 | 0.1483 | 0.2984 |

### Result interpretation

The baseline model achieved high precision but very low recall. It detected only a small proportion of the annotated defect regions.

After changing the training configuration and augmentation settings, the improved model achieved substantially better recall and mAP@0.5.

The improved model's recall increased from 0.16 to 0.52, while mAP@0.5 increased from 0.1629 to 0.4698.

However, precision decreased from 0.9490 to 0.5914. This indicates that the improved model detected more defect regions but also produced more incorrect detections.

For this application, missing an actual defect can be costly. Therefore, recall is an important metric, but the false-positive rate must also be considered before deployment.

The improved model is a better experimental candidate than the baseline, but it is not yet ready for production use.

---

## 5. Failure Case Analysis

I inspected the improved model's predictions on the test images and selected six representative failure cases.

The analysis used a confidence threshold of 0.25 and an IoU threshold of 0.5 for greedy one-to-one matching between predictions and ground-truth boxes.

The counts below come from this separate diagnostic matching script. They are not the same as the aggregate Ultralytics mAP calculation.

### Failure case results

| Image | Ground-truth boxes | Predicted boxes | Missed boxes | False positives |
|---|---:|---:|---:|---:|
| bent_000.png | 2 | 1 | 1 | 0 |
| bent_018.png | 2 | 1 | 1 | 0 |
| color_003.png | 7 | 1 | 7 | 1 |
| scratch_001.png | 1 | 3 | 0 | 2 |
| scratch_006.png | 3 | 2 | 2 | 1 |
| scratch_012.png | 2 | 3 | 2 | 3 |

### Visual failure cases

The following images are included in the repository under `reports/failure_cases/`.

**1. Bent defect – missed region**

![Bent defect failure case](reports/failure_cases/failure_1_bent_000.png)

The model detected one region but missed another annotated region, suggesting that it may struggle to localize multiple defect regions in the same image.

**2. Bent defect – missed region**

![Bent defect failure case](reports/failure_cases/failure_2_bent_018.png)

One of the two annotated defect regions was missed, indicating that detection performance can vary across images from the same defect category.

**3. Color defect – poor localization**

![Color defect failure case](reports/failure_cases/failure_3_color_003.png)

The image contains several annotated regions, but the model produced only one prediction, with no successful match under the diagnostic IoU criterion.

**4. Scratch defect – false-positive predictions**

![Scratch defect failure case](reports/failure_cases/failure_4_scratch_001.png)

The model detected the annotated region but also generated two additional predictions that did not match the ground truth.

**5. Scratch defect – missed regions**

![Scratch defect failure case](reports/failure_cases/failure_5_scratch_006.png)

The model missed two of the three annotated regions, showing difficulty in detecting all defect areas.

**6. Scratch defect – missed regions and false positives**

![Scratch defect failure case](reports/failure_cases/failure_6_scratch_012.png)

The model produced additional incorrect detections while also missing annotated defect regions, indicating inconsistent localization.

### What I learned from the failure cases

The failure analysis suggests that the model still has difficulty with:

- Detecting multiple small defect regions in one image.
- Localizing irregular or fragmented defect regions.
- Avoiding false positives on scratch-like patterns.
- Maintaining consistent detection performance across different defect appearances.

These observations are based on the selected test examples. Further experiments with more data and a larger, independently held-out test set would be required to confirm the underlying causes.

---

## 6. ONNX Export and CPU Inference Benchmark

After training, I exported the improved YOLOv8n model to ONNX format.

ONNX makes it possible to use the trained model with inference runtimes outside the original training framework.

### Export configuration

| Parameter | Value |
|---|---|
| Model | Improved YOLOv8n |
| Export format | ONNX |
| Precision | FP32 |
| ONNX opset | 17 |
| Batch size | 1 |
| Dynamic input | Disabled |
| Simplification | Enabled |
| Execution provider | ONNX Runtime CPUExecutionProvider |

**Available exported model:** `models/improved_v2_320.onnx`

The model was exported with an input resolution of 320 × 320.

### CPU benchmark methodology

I measured inference time using ONNX Runtime on my laptop CPU.

The benchmark used:

- One input image tensor at a time.
- Batch size of 1.
- Five warm-up runs.
- Thirty timed inference runs.
- ONNX Runtime CPU execution provider.

The benchmark measured model execution using a prepared input tensor. It does not represent the complete image-processing pipeline, including image loading, resizing, normalization, and application-level post-processing.

### Inference benchmark results

| Input resolution | Mean latency | Median latency | P95 latency |
|---|---:|---:|---:|
| 320 × 320 | 25.71 ms | 23.29 ms | 51.48 ms |
| 256 × 256 | 18.98 ms | 14.32 ms | 35.86 ms |

The 256 × 256 result was obtained by exporting the same trained checkpoint at a smaller input resolution. It was not a separately retrained model.

### Accuracy and speed comparison

| Metric | 320 × 320 | 256 × 256 |
|---|---:|---:|
| Precision | 0.5914 | 0.6288 |
| Recall | 0.5200 | 0.4400 |
| mAP@0.5 | 0.4698 | 0.4128 |
| mAP@0.5:0.95 | 0.2984 | 0.2597 |
| Mean ONNX latency | 25.71 ms | 18.98 ms |

The smaller input resolution reduced mean inference latency by approximately 26.2%, but it also reduced recall and mAP@0.5.

This shows the trade-off between speed and detection performance. The 256 × 256 setting was faster, while the 320 × 320 setting provided better detection results in this experiment.

### Hardware details

- CPU: 12th Gen Intel Core i5-1235U
- Operating system: Windows
- Python: 3.11.9
- PyTorch: 2.14.1+cpu
- ONNX Runtime: 1.31.1

### Does the model meet the 10 ms requirement?

No. Neither tested resolution achieved a mean inference time below 10 ms on this CPU.

The 256 × 256 configuration was faster, but its measured mean latency was still 18.98 ms.

This means further optimization or a different deployment strategy would be required to meet a strict 10 ms target.

---

## 7. Improvement Experiment

I conducted two related experiments to understand how training configuration and inference resolution affect performance.

### Experiment 1: Training configuration changes

I changed the optimizer to AdamW, adjusted the learning rate, introduced a warm-up period, and modified the augmentation settings.

**Result:**

- Recall improved from 0.16 to 0.52.
- mAP@0.5 improved from 0.1629 to 0.4698.
- Precision decreased from 0.9490 to 0.5914.

**Conclusion:** The changes improved the model's ability to detect more defect regions, but increased false-positive predictions. Overall, this was a useful improvement, although further tuning is needed.

### Experiment 2: Reducing input resolution

I exported and evaluated the same improved checkpoint at 256 × 256 instead of 320 × 320.

**Result:**

- Mean ONNX latency decreased from 25.71 ms to 18.98 ms.
- mAP@0.5 decreased from 0.4698 to 0.4128.
- Recall decreased from 0.52 to 0.44.

**Conclusion:** Reducing input resolution improved speed but sacrificed some detection performance. It did not achieve the required 10 ms target.

---

## 8. Written Questions

### A. If you had only 30 defective images and 3,000 good images, how would you design a reliable system?

With such an imbalanced dataset, I would not rely only on a supervised object detection model trained on the available defective images.

I would approach the problem as an anomaly detection task.

**My approach would be:**

1. **Learn normal behaviour:** Train an anomaly detection model primarily using the 3,000 good images. The model would learn the normal appearance and structure of a metal nut.

2. **Use the defective images for validation:** Keep the 30 defective images separate where possible and use them to evaluate whether the system can identify abnormal regions.

3. **Apply anomaly detection:** Explore methods such as PatchCore or similar feature-based anomaly detection approaches. These can learn normal visual patterns without requiring a large number of labelled defect examples.

4. **Use augmentation carefully:** Apply realistic transformations to the limited defective images, such as small rotations, brightness changes, and translations. I would avoid transformations that create unrealistic defects.

5. **Set a suitable threshold:** Choose an anomaly threshold based on validation data, with particular attention to avoiding missed defects.

6. **Monitor performance:** Track false negatives and false positives separately. In industrial inspection, missed defects can be especially costly.

7. **Collect more data over time:** Store reviewed defect examples and use them to improve the system as more real production data becomes available.

**My recommendation:** Start with an anomaly detection approach trained on normal images, then compare it with a supervised detector as more defect examples become available.

I would also evaluate performance on a separate test set containing examples from different production conditions.

### B. If the model starts missing defects two weeks after deployment, what would you check first and why?

I would investigate the issue systematically rather than immediately retraining the model.

My checks would be:

**1. Verify the input data and camera**

First, I would check whether the camera position, lighting, focus, exposure, or image resolution has changed.

A change in image quality can make the production images different from the images used during training.

**2. Check preprocessing**

I would verify that resizing, normalization, color conversion, and image handling are consistent with the training and deployment pipeline.

An incorrect preprocessing change can affect model predictions even when the model itself has not changed.

**3. Review confidence scores and thresholds**

I would inspect whether the model is detecting defects with lower confidence than before.

A threshold that was suitable during testing may not be suitable for new production conditions.

**4. Inspect false negatives**

I would collect recently missed defect images and compare them with the training examples.

This would help identify whether the model is struggling with new defect shapes, smaller defects, different materials, or different lighting conditions.

**5. Check data drift**

I would compare recent production images with the original training data to identify changes in image appearance or defect patterns.

**6. Check model and software versions**

I would verify that the correct model file, inference code, configuration, and preprocessing version are deployed.

**7. Retrain only after identifying the cause**

If the new images represent a genuine change in production conditions, I would collect and label representative examples, update the training dataset, retrain or fine-tune the model, and validate it before deployment.

I would also maintain a rollback option in case the updated model performs worse.

**My priority would be:** Check the camera and input pipeline first, then thresholds and missed predictions, followed by data drift and model updates.

### C. If the controller allows only 10 ms per image but the model takes 40 ms, what would you do?

If the system has a strict 10 ms inference limit and the current model takes 40 ms, I would consider several options.

**1. Reduce input resolution**

A smaller input image can reduce computation. I tested this approach by reducing the input resolution from 320 × 320 to 256 × 256.

It reduced mean ONNX latency from 25.71 ms to 18.98 ms, but did not meet the 10 ms target.

**2. Use a smaller model**

I would compare smaller architectures or a more lightweight detection model. The aim would be to reduce computation while maintaining acceptable defect detection performance.

**3. Quantization**

I would test INT8 quantization to reduce model size and potentially improve inference speed. However, quantization does not always make a model faster on every CPU, so I would benchmark it on the actual target hardware.

**4. Use an optimized inference runtime**

I would benchmark alternatives such as OpenVINO and other hardware-supported inference runtimes.

In my experiments, the OpenVINO baseline export was not faster on my current CPU, so I would not assume that changing runtimes automatically improves latency.

**5. Profile the complete pipeline**

I would measure image acquisition, preprocessing, model inference, and post-processing separately. The actual bottleneck may not be the model alone.

**6. Consider hardware acceleration**

If the hardware budget allows it, I would evaluate a supported accelerator or a more capable edge device.

**7. Use a two-stage inspection approach**

If appropriate for the application, a lightweight first-stage model could identify suspicious regions, followed by more detailed inspection only when necessary.

**My conclusion:** I would first profile the full pipeline, then benchmark a smaller model, lower resolution, quantization, and hardware-specific runtimes. I would select the solution based on both latency and defect detection performance, not speed alone.

---

## 9. Limitations and Future Improvements

The project demonstrates the complete workflow from mask conversion to model training, evaluation, failure analysis, and ONNX benchmarking. However, there are limitations that should be addressed before production use.

### Current limitations

- The dataset is small, with only 93 defective images.
- The model uses one defect class and does not classify individual defect types.
- The custom split is not an untouched official MVTec test split.
- The test set contains only 36 images, so the metrics may vary significantly with a different split.
- The improved model still misses some annotated defect regions.
- The model does not currently meet the 10 ms mean inference requirement on my CPU.
- The CPU benchmark measures model execution on a prepared tensor rather than the complete production pipeline.

### Future improvements

1. Collect more defective metal nut images under different lighting and production conditions.
2. Evaluate anomaly detection methods such as PatchCore.
3. Experiment with higher-quality bounding-box annotations and different minimum component-area thresholds.
4. Investigate whether separate defect classes improve the usefulness of the model.
5. Tune confidence and IoU thresholds using a validation set.
6. Test additional lightweight detection architectures.
7. Benchmark inference on the actual target deployment hardware.
8. Evaluate the complete image acquisition-to-decision latency.
9. Use a separate, untouched test set for final performance evaluation.

---

## 10. AI Tool Usage Disclosure

AI tools were used as development assistance during this assignment.

They helped with understanding implementation options, debugging code, interpreting evaluation results, and improving documentation.

The dataset preparation, training experiments, model evaluation, failure-case inspection, and inference benchmarking were carried out through the project code and local execution.

I reviewed the implementation and results and can explain the technical decisions and experimental outcomes.

---

## 11. Conclusion

Through this assignment, I implemented a complete experimental object detection workflow for metal nut defect detection.

I started by converting pixel-level defect masks into YOLO bounding-box annotations, prepared the dataset, trained a YOLOv8n baseline, and then experimented with an improved training configuration.

The improved model achieved a recall of 0.52 and mAP@0.5 of 0.4698 on the custom test split, compared with 0.16 recall and 0.1629 mAP@0.5 for the baseline.

I also inspected six failure cases to understand where the model struggles, exported the trained model to ONNX, and measured CPU inference latency at two input resolutions.

The experiments showed that improving recall can increase false positives, while reducing input resolution can improve speed at the cost of detection accuracy. Neither tested configuration met the 10 ms mean inference target, so further optimization is required.

My main takeaway is that building an industrial defect detection system is not just about training a model. Dataset quality, annotation strategy, failure analysis, inference speed, and deployment conditions are equally important.

I would be interested in improving this system further with additional production data, anomaly detection techniques, and deployment-focused optimization.

---

## 12. Project Files and Reproduction

The GitHub repository should contain the following project components:

| Component | Location |
|---|---|
| Mask-to-box conversion | `src/convert_masks.py` |
| Baseline training | `src/train.py` |
| Improved training | `src/train_improved.py` |
| Dataset configuration | `configs/metal_nut.yaml` |
| Failure-case inspection | `src/inspect_failures_v2.py` |
| Failure-case images | `reports/failure_cases/` |
| ONNX model (320 × 320) | `models/improved_v2_320.onnx` |
| Project report | `reports/assignment_writeup.md` |

**Repository:** [Add your public GitHub repository URL here]

### Basic reproduction steps

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd insiso-metal-nut-inspection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run mask conversion:

```bash
python src/convert_masks.py
```

Train the baseline model:

```bash
python src/train.py
```

Train the improved model:

```bash
python src/train_improved.py
```

The dataset must be downloaded separately from the [MVTec AD dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and placed in the expected `data/raw/metal_nut` directory before running the conversion and training scripts.

The trained checkpoints and exported ONNX model should be included in the repository or made available through a clearly documented download link. Large generated files may be provided separately if repository size limits make committing them impractical.

---

**End of assignment write-up**
