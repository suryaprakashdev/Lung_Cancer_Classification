# LIDC-IDRI Lung Nodule Detection — Full Architecture Plan

## Project Goal

Build an end-to-end pipeline where anyone can upload a CT DICOM folder and a CNN will identify nodules, classify their malignancy, and visualise the results with 3D GradCAM heatmaps.

---

## Dataset

**Source:** NCI Imaging Data Commons (IDC) — LIDC-IDRI collection
**Size:** ~110 GB, 1,018 patients
**Access:** Public AWS S3 buckets (free egress) via s5cmd manifest

**Per patient structure:**

```
LIDC-IDRI-XXXX/
  CT series/      → 3D lung scan (hundreds of .dcm slices)
  SEG series/     → nodule contour masks (one per radiologist, up to 4)
  SR series/      → malignancy scores + measurements per radiologist
```

**Labels:**

- Up to 4 radiologists annotated each scan independently
- Malignancy scored 1–5 per radiologist per nodule
- Consensus: average score, threshold at 3 (≥3 = malignant)
- Nodule matching across radiologists: centroid distance ≤5mm

---

## Infrastructure

| Service                  | Purpose                               | Cost estimate   |
| ------------------------ | ------------------------------------- | --------------- |
| IDC Public S3            | Source DICOM data                     | Free            |
| Azure Blob Storage       | Zarr volumes + model weights + labels | ~£2/month       |
| Azure ML CPU cluster     | Preprocessing job (one-time)          | ~£5 one-time    |
| Azure ML GPU cluster     | Full training runs                    | ~£10–20 per run |
| Azure Container Registry | Docker images                         | Free tier       |
| Azure Container Apps     | BentoML inference API                 | Pay per request |
| University GPU Server    | Development + experiments             | Free            |
| Local Mac + VS Code      | Development                           | Free            |

---

## Phase 1 — Data Preparation

**Status:** DICOM data currently uploading to Azure Blob Storage via parallel ingestion script.

**Blob structure on completion:**

```
lidc-idri/
  LIDC-IDRI-0001/
    1.3.6.1.4.1.14519.../ → CT slices (.dcm)
    1.2.276.0.7230010.../ → SEG files (.dcm)
    1.2.276.0.7230010.../ → SR files (.dcm)
  LIDC-IDRI-0002/
  ...
```

**Also available:** s5cmd manifest from IDC portal pointing directly to all S3 URLs — used in preprocessing to stream data without re-downloading from blob.

---

## Phase 2 — Preprocessing

**Where it runs:** Azure ML CPU cluster (parallel workers, one-time job)

**Input:** s5cmd manifest (S3 URLs) + SEG/SR DICOM files

**Process per patient:**

### 2a — CT Volume

```
Stream CT DICOM from S3
  → sort slices by InstanceNumber
  → stack into 3D numpy array (D × 512 × 512)
  → apply HU windowing (lung window: -1000 to 400)
  → normalise to float32 [0, 1]
  → save as Zarr → blob: processed/volumes/LIDC-IDRI-XXXX.zarr
```

**Zarr chunk strategy:**

- Full volume storage: `(D, 512, 512)` — one chunk per axial slice
- Training patch extraction at read time from full volume

### 2b — Nodule Masks

```
Stream SEG DICOM files from S3
  → parse nodule contours per radiologist (pydicom)
  → match nodules across radiologists (centroid ≤5mm threshold)
  → consensus mask: ≥2 of 4 radiologists must agree
  → save as Zarr → blob: processed/masks/LIDC-IDRI-XXXX.zarr
```

### 2c — Labels

```
Stream SR DICOM files from S3
  → extract malignancy scores (1–5 per radiologist per nodule)
  → average scores across radiologists
  → threshold: mean ≥3 = malignant (1), <3 = benign (0)
  → keep individual scores for uncertainty analysis
  → save to dataset.json (MONAI standard format)
```

**Output in blob:**

```
processed/
  volumes/
    LIDC-IDRI-0001.zarr
    LIDC-IDRI-0002.zarr
    ...
  masks/
    LIDC-IDRI-0001.zarr
    LIDC-IDRI-0002.zarr
    ...
  dataset.json
```

**dataset.json structure per entry:**

```json
{
  "image": "processed/volumes/LIDC-IDRI-0001.zarr",
  "mask": "processed/masks/LIDC-IDRI-0001.zarr",
  "nodules": [
    {
      "centroid": [x, y, z],
      "malignancy": 1,
      "mean_score": 3.5,
      "radiologist_scores": [3, 4, 3, 4],
      "size_mm": 6.2
    }
  ]
}
```

---

## Phase 3 — Model Training

**Development runs:** University GPU server (free, VS Code Remote SSH)
**Full training runs:** Azure ML GPU cluster (`Standard_NC6s_v3` — V100 16GB)

**Experiment tracking:** Azure ML — logs loss curves, FROC/AUC per epoch, hyperparameters, checkpoints

**Train/val/test split:** 80/10/10 patient-level split (never mix patients across splits)

---

### Model 1 — 3D UNet (Nodule Detector)

**Task:** Given a full CT volume, produce a 3D probability heatmap of nodule locations

**Architecture:** MONAI `UNet` — 3D, 5 levels, residual units

**Input:** Full normalised CT volume `(1 × D × 512 × 512)`

**Output:** 3D probability heatmap, same spatial dimensions as input

**MONAI Transforms:**

```python
ScaleIntensityRanged(a_min=-1000, a_max=400)
RandCropByPosNegLabeld(spatial_size=(96, 96, 96))
RandFlipd(prob=0.5)
RandRotate90d(prob=0.5)
ToTensord()
```

**Loss:** Dice Loss + Binary Cross Entropy (combined)

**Metric:** FROC (Free-Response ROC) — standard for nodule detection

**Output:** Candidate nodule centroids above probability threshold (0.5)

**Saves to blob:**

```
models/detector/
  unet_best.pth
  unet_config.json
```

---

### Model 2 — 3D ResNet (Malignancy Classifier)

**Task:** Given a 64³ patch around a nodule candidate, classify as malignant / benign / false positive

**Architecture:** MONAI `DenseNet121` adapted for 3D — or `ResNet` (preferred for GradCAM)

**Why ResNet over DenseNet:** GradCAM gradient flow is cleaner through residual connections. DenseNet's dense connections dilute gradient signals producing noisier activation maps.

**Input:** 64 × 64 × 64 patch centred on nodule candidate

**Output:** Malignancy probability (0–1)

**MONAI Transforms:**

```python
RandFlipd(prob=0.5)
RandRotate90d(prob=0.5)
RandGaussianNoised(prob=0.2)
ScaleIntensityRanged(a_min=-1000, a_max=400)
ToTensord()
```

**Loss:** Binary Cross Entropy with class weights (dataset is imbalanced — more benign than malignant)

**Metric:** AUC-ROC, sensitivity at fixed specificity

**Saves to blob:**

```
models/classifier/
  resnet_best.pth
  resnet_config.json
```

---

### GradCAM 3D Visualisation

**Library:** `captum` (PyTorch) — GradCAM with 3D support

**Target layer:** Last convolutional layer of ResNet classifier

**Output:** 3D activation heatmap, same shape as input patch (64 × 64 × 64)

**Interpretation:** High activation regions = voxels most responsible for malignancy prediction

**Display:** Overlaid as colour map on axial/coronal/sagittal slices in frontend

---

## Phase 4 — Inference Pipeline

```
User uploads DICOM folder/zip
        │
        ▼
Inline Preprocessing
  → read DICOM slices
  → sort by InstanceNumber
  → stack into 3D volume
  → HU window + normalise
        │
        ▼
BentoML Service
        │
        ├── Runner 1: UNet Detector
        │     → 3D probability heatmap
        │     → threshold at 0.5
        │     → connected components
        │     → candidate centroids + bounding boxes
        │
        └── Runner 2: ResNet Classifier
              For each candidate:
              → extract 64³ patch
              → malignancy probability
              → GradCAM 3D heatmap
              → confidence score
        │
        ▼
Output per nodule:
  {
    "location": {"x": int, "y": int, "z": int},
    "size_mm": float,
    "malignancy_probability": float,
    "confidence": float,
    "label": "malignant" | "benign" | "false_positive",
    "gradcam_heatmap": "path/to/heatmap.npy"
  }
```

---

## Phase 5 — BentoML Service

**Why BentoML over FastAPI:**

- Runner architecture handles two models in sequence natively
- Adaptive batching built in — multiple concurrent users batched automatically
- Model versioning and storage handled natively
- One command deployment to Azure Container Apps

**Service structure:**

```python
# Two runners
detector_runner   = UNetDetector.to_runner()
classifier_runner = ResNetClassifier.to_runner()

# One service orchestrating both
svc = bentoml.Service(
    "nodule_detector",
    runners=[detector_runner, classifier_runner]
)

# Endpoints
POST /predict    → full pipeline, returns nodule list
GET  /health     → liveness check
GET  /metrics    → inference latency, throughput
```

**Model store:** BentoML model store backed by Azure Blob Storage

**Deployment:**

```
bento build
  → Docker image
  → push to Azure Container Registry
  → deploy to Azure Container Apps
```

**GPU caveat:** Azure Container Apps student tier is CPU only. Options:

- Accept slower inference (~2–3 min per scan on CPU) — fine for a research demo
- Use Azure ML managed endpoints for GPU-backed inference (higher cost)

---

## Phase 6 — Frontend

**Stack:** React + cornerstone.js (medical imaging viewer)

**Hosted on:** Azure Static Web Apps (free tier)

**User flow:**

```
1. Upload DICOM folder or zip
2. Loading screen while pipeline runs
3. Results page:
   └── 3D CT viewer with nodule markers overlaid
   └── Nodule list (sorted by malignancy probability)
   └── Click nodule →
         Axial / coronal / sagittal slices
         GradCAM heatmap overlay (heat colour map)
         Malignancy probability bar
         Size in mm
         Confidence score
```

---

## Development Timeline

| Week | Task                                                                   |
| ---- | ---------------------------------------------------------------------- |
| 1    | Preprocessing script — S3 → Zarr → blob, dataset.json generation       |
| 2    | Label sanity checks — verify masks, score distributions, class balance |
| 3    | UNet detector — train on university GPU, validate FROC                 |
| 4    | ResNet classifier — train on university GPU, validate AUC-ROC          |
| 5    | Full training run on Azure ML GPU — hyperparameter tuning              |
| 6    | BentoML service — wire both models, GradCAM integration                |
| 7    | Deploy to Azure Container Apps                                         |
| 8    | Frontend — CT viewer, heatmap overlay, results UI                      |

---

## Key Libraries

| Library              | Purpose                                             |
| -------------------- | --------------------------------------------------- |
| `monai`              | Medical imaging transforms, UNet, dataset utilities |
| `pydicom`            | DICOM file parsing (CT slices, SEG, SR)             |
| `zarr`               | Chunked array storage format                        |
| `azure-storage-blob` | Read/write blob storage                             |
| `idc-index`          | IDC metadata + S3 URL lookup                        |
| `torch`              | Model training                                      |
| `captum`             | GradCAM 3D visualisation                            |
| `bentoml`            | Model serving + API                                 |
| `s5cmd`              | Fast S3 manifest downloads                          |
| `numpy`, `scipy`     | Array operations, connected components              |
| `rich`               | CLI progress and logging                            |

---

## Repository Structure

```
Lung_Cancer_Classification/
  ├── utils/
  │     └── lidc_to_azure_blob.py     ← current ingestion script
  │
  ├── preprocessing/
  │     ├── dicom_to_zarr.py          ← CT volume conversion
  │     ├── parse_seg.py              ← nodule mask extraction
  │     ├── parse_sr.py               ← malignancy label extraction
  │     └── build_dataset_json.py     ← MONAI dataset.json builder
  │
  ├── training/
  │     ├── dataset.py                ← MONAI Dataset + DataLoader
  │     ├── transforms.py             ← augmentation pipeline
  │     ├── train_detector.py         ← UNet training script
  │     ├── train_classifier.py       ← ResNet training script
  │     └── evaluate.py               ← FROC + AUC-ROC evaluation
  │
  ├── inference/
  │     ├── preprocess.py             ← inline DICOM preprocessing
  │     ├── detector.py               ← UNet inference
  │     ├── classifier.py             ← ResNet inference + GradCAM
  │     └── pipeline.py               ← chain detector + classifier
  │
  ├── serving/
  │     ├── service.py                ← BentoML service definition
  │     ├── bentofile.yaml            ← bento build config
  │     └── Dockerfile
  │
  ├── frontend/
  │     └── (React app)
  │
  ├── logs/
  │     └── lidc_ingest.log
  │
  ├── .env
  └── README.md
```

---

## Success Criteria

- Anyone can upload a CT DICOM folder via the web UI
- Pipeline identifies nodule locations automatically
- Each nodule returns a malignancy probability with confidence score
- GradCAM heatmap shows which regions drove the prediction
- Results displayed on interactive CT viewer with slice-level overlays
