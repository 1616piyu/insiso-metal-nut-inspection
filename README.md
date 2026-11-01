# Metal Nut Defect Detection – InSiSo Technologies Assignment

A lightweight industrial defect detection project using **YOLOv8n** and the **MVTec Anomaly Detection (MVTec AD) metal_nut dataset**.

The project covers dataset preparation, pixel-mask to bounding-box conversion, model training, evaluation, failure-case analysis, ONNX export, and CPU inference benchmarking.

## 1. Project Overview

The objective is to identify defective regions in metal nut images.

The dataset contains four defect categories:

- Bent
- Color
- Flip
- Scratch

I converted the original pixel-level defect masks into YOLO bounding-box annotations and trained a YOLOv8n object detection model.

**Note:** The current detector uses a single class, `defect`. It localizes defective regions but does not separately classify the four defect categories.

## 2. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Development |
| PyTorch | Deep learning framework |
| Ultralytics YOLOv8n | Object detection |
| OpenCV | Image and mask processing |
| NumPy | Numerical operations |
| ONNX | Model export |
| ONNX Runtime | CPU inference |
| scikit-learn | Supporting evaluation utilities |

## 3. Project Workflow

1. Dataset preparation
2. Pixel-mask to bounding-box conversion
3. YOLO dataset preparation
4. Baseline YOLOv8n training
5. Improved YOLOv8n training
6. Test-set evaluation
7. Failure-case analysis
8. ONNX model export
9. CPU inference benchmarking

## 4. Dataset

**Dataset:** [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

The project uses the `metal_nut` category.

### Dataset summary

| Category | Images |
|---|---:|
| Train – Good | 220 |
| Test – Good | 22 |
| Test – Bent | 25 |
| Test – Color | 22 |
| Test – Flip | 23 |
| Test – Scratch | 23 |
| **Total** | **335** |

The dataset contains 93 defective images with corresponding pixel-level masks.

### Dataset setup

Download the MVTec AD dataset and extract the `metal_nut` category into:

```text
data/raw/metal_nut/
```

The expected structure is:

```text
data/
└── raw/
    └── metal_nut/
        â”œ── train/
        │   └── good/
        â”œ── test/
        │   â”œ── good/
        │   â”œ── bent/
        │   â”œ── color/
        │   â”œ── flip/
        │   └── scratch/
        └── ground_truth/
            â”œ── bent/
            â”œ── color/
            â”œ── flip/
            └── scratch/
```

The original dataset is not included in this repository. Download it separately and place it in the expected directory before running the data preparation scripts.

## 5. Mask-to-Bounding-Box Conversion

The original dataset provides pixel-level segmentation masks. Since YOLO object detection requires bounding boxes, I implemented a conversion script:

```text
src/convert_masks.py
```

The script:

1. Reads defect masks in grayscale.
2. Converts masks into binary images.
3. Uses OpenCV connected component analysis to identify separate defect regions.
4. Calculates bounding boxes for individual components.
5. Filters components smaller than 10 pixels.
6. Normalizes coordinates into YOLO format.
7. Saves the annotations as text files.

### Annotation format

```text
class_id x_center y_center width height
```

The project uses one class:

```text
0 = defect
```

### Conversion results

| Metric | Result |
|---|---:|
| Defective images processed | 93 |
| Bounding boxes generated | 132 |
| Invalid annotations | 0 |
| Minimum component area | 10 pixels |

Normal images have empty annotation files because no defect regions are present.

## 6. Model Training

Two training experiments were performed using pretrained YOLOv8n weights.

### Baseline model

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Image size | 320 × 320 |
| Maximum epochs | 30 |
| Batch size | 8 |
| Seed | 42 |
| Early stopping patience | 8 |

Training script:

```text
src/train.py
```

### Improved model

The second experiment used AdamW, a lower initial learning rate, warm-up, and modified data augmentation settings.

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Image size | 320 × 320 |
| Maximum epochs | 60 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0005 |
| Warm-up epochs | 2 |
| Seed | 42 |
| Early stopping patience | 15 |

Training script:

```text
src/train_improved.py
```

The improved training run stopped early after 41 epochs.

## 7. Evaluation Results

Both models were evaluated on the same custom test split.

| Metric | Baseline YOLOv8n | Improved YOLOv8n |
|---|---:|---:|
| Precision | 0.9490 | 0.5914 |
| Recall | 0.1600 | 0.5200 |
| mAP@0.5 | 0.1629 | 0.4698 |
| mAP@0.5:0.95 | 0.1483 | 0.2984 |

### Observations

The improved training configuration increased recall from 0.16 to 0.52 and mAP@0.5 from 0.1629 to 0.4698.

However, precision decreased from 0.9490 to 0.5914, indicating an increase in incorrect detections.

The improved model detects more defect regions than the baseline, but further work is needed to reduce false positives and missed defects.

## 8. Failure-Case Analysis

I inspected the improved model's predictions and selected six representative failure cases.

The images and diagnostic results are available below.

| Failure case | Image |
|---|---|
| Bent defect – missed region | [View image](reports/failure_cases/failure_1_bent_000.png) |
| Bent defect – missed region | [View image](reports/failure_cases/failure_2_bent_018.png) |
| Color defect – poor localization | [View image](reports/failure_cases/failure_3_color_003.png) |
| Scratch defect – false positives | [View image](reports/failure_cases/failure_4_scratch_001.png) |
| Scratch defect – missed regions | [View image](reports/failure_cases/failure_5_scratch_006.png) |
| Scratch defect – missed regions and false positives | [View image](reports/failure_cases/failure_6_scratch_012.png) |

### Main observations

- Some images contain multiple defect regions, but the model detects only a subset.
- Small or fragmented regions can be difficult to localize.
- Scratch-like patterns can produce false-positive detections.
- Performance varies across defect appearances.

These examples show why failure-case inspection is important in addition to reporting aggregate evaluation metrics.

## 9. ONNX Export and CPU Benchmark

The improved YOLOv8n model was exported to ONNX format and benchmarked using ONNX Runtime with the CPU execution provider.

### Hardware

- CPU: 12th Gen Intel Core i5-1235U
- Operating system: Windows
- Python: 3.11.9
- ONNX Runtime: 1.31.1

### Benchmark results

| Input resolution | Mean latency | Median latency | P95 latency |
|---|---:|---:|---:|
| 320 × 320 | 25.71 ms | 23.29 ms | 51.48 ms |
| 256 × 256 | 18.98 ms | 14.32 ms | 35.86 ms |

The 256 × 256 result uses the same trained checkpoint exported at a smaller input resolution. It is not a separately trained model.

The benchmark uses five warm-up runs and 30 timed inference runs. It measures model execution on a prepared input tensor and excludes image loading, preprocessing, and application-level post-processing.

### Observation

Reducing input resolution from 320 × 320 to 256 × 256 reduced mean ONNX inference latency by approximately 26.2%.

However, the smaller input resolution also reduced recall and mAP@0.5.

Neither configuration achieved the 10 ms mean inference target on the tested CPU.

### ONNX model

The 320 × 320 ONNX export is available at:

```text
models/improved_v2_320.onnx
```

If the model is not committed to GitHub, provide a separate download link here.

## 10. Improvement Experiments

Two experiments were conducted.

### Experiment 1: Improved training configuration

Changed the optimizer to AdamW, adjusted learning-rate settings, and modified augmentation.

**Outcome:** Recall and mAP@0.5 improved, although precision decreased.

### Experiment 2: Smaller inference resolution

Exported the same trained model at 256 × 256 instead of 320 × 320.

**Outcome:** Mean latency decreased from 25.71 ms to 18.98 ms, but recall and mAP@0.5 also decreased.

These experiments demonstrate the trade-off between defect detection performance and inference speed.

## 11. Limitations

- The dataset contains a limited number of defective images.
- The detector uses one defect class rather than four separate defect classes.
- The custom evaluation split is small and was reused during experimentation.
- The custom split is not an untouched official MVTec test benchmark.
- The model still misses some annotated defect regions.
- The tested CPU configurations did not achieve the 10 ms mean latency target.
- The ONNX benchmark excludes the complete image-processing pipeline.

Therefore, the reported results should be considered preliminary and the model should not be treated as production-ready.

## 12. Future Improvements

- Collect more defective images under different production conditions.
- Experiment with anomaly detection approaches such as PatchCore.
- Improve defect localization and annotation quality.
- Evaluate separate defect classes if sufficient labelled data becomes available.
- Tune confidence thresholds using validation data.
- Test smaller detection architectures and quantization.
- Benchmark on the actual target deployment hardware.
- Evaluate the complete image-to-decision pipeline.

## 13. Written Assignment Answers

The detailed responses to the three written questions are included in:

[**Assignment Technical Report – reports/assignment_writeup.md**](reports/assignment_writeup.md)

The report covers:

- Designing a reliable system with very few defective images.
- Investigating missed defects after deployment.
- Meeting a strict 10 ms inference requirement.

## 14. AI Tool Usage Disclosure

AI tools were used as development assistance for understanding implementation options, debugging, interpreting results, and improving documentation.

The dataset preparation, training experiments, evaluation, failure-case inspection, and CPU benchmarking were performed through the project's code and local execution.

I reviewed the implementation and results and can explain the technical decisions and experiments.

## 15. Repository Structure

The main project components are organized as follows:

```text
insiso-metal-nut-inspection/
│
â”œ── configs/
│   └── metal_nut.yaml
│
â”œ── data/
│   â”œ── raw/
│   │   └── metal_nut/
│   â”œ── processed/
│   └── yolo_rebuilt/
│
â”œ── reports/
│   â”œ── assignment_writeup.md
│   └── failure_cases/
│       â”œ── failure_1_bent_000.png
│       â”œ── failure_2_bent_018.png
│       â”œ── failure_3_color_003.png
│       â”œ── failure_4_scratch_001.png
│       â”œ── failure_5_scratch_006.png
│       └── failure_6_scratch_012.png
│
â”œ── src/
│   â”œ── convert_masks.py
│   â”œ── train.py
│   â”œ── train_improved.py
│   └── inspect_failures_v2.py
│
â”œ── requirements.txt
â”œ── README.md
└── .gitignore
```

Additional evaluation and export scripts, model weights, and benchmark outputs should be included in the repository where applicable.

## 16. Setup and Reproduction

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd insiso-metal-nut-inspection
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download and prepare the dataset

Download the MVTec AD dataset from:

https://www.mvtec.com/company/research/datasets/mvtec-ad

Extract the `metal_nut` category into:

```text
data/raw/metal_nut/
```

### Step 5: Convert masks to bounding boxes

```bash
python src/convert_masks.py
```

### Step 6: Train the baseline model

```bash
python src/train.py
```

### Step 7: Train the improved model

```bash
python src/train_improved.py
```

The evaluation, failure-analysis, and ONNX export scripts are available under `src/`. Use the corresponding scripts and configurations documented in the technical report.

---

## Author

**Computer Vision / AI-ML Internship Assignment**
**Organization:** InSiSo Technologies
