# Lung Cancer Nodule Classification — LIDC-IDRI

Binary malignancy classification of lung nodules from the LIDC-IDRI CT dataset.
Compares a custom CNN against three ImageNet-pretrained architectures, with
Grad-CAM visualisation for interpretability.

---

## Project structure

```
lung_cancer_project/
├── preprocessing.py    # DICOM → PNG patch extraction (Stage 1)
├── dataset.py          # DataLoader factory (used by train & postprocessing)
├── models.py           # All model definitions
├── train.py            # Training + validation loop (Stage 2)
├── postprocessing.py   # Test evaluation, Grad-CAM, ROC curves (Stage 3)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **NumPy version**: `requirements.txt` pins NumPy to `1.26.4` because
> `pylidc` monkey-patches deprecated aliases (`np.int`, `np.float`, `np.bool`)
> that were removed in NumPy 2.x.

---

## Data

1. Download the LIDC-IDRI dataset from
   [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/).
2. Unzip/place all DICOM files under a single root directory,
   e.g. `/data/LIDC-IDRI/`.

---

## Run order

### Stage 1 — Preprocessing

Converts raw DICOMs → labelled PNG nodule patches under `Benign_0/` and
`Malignant_1/` subdirectories. Supports crash-safe resumption via a
checkpoint file.

```bash
python preprocessing.py \
    --raw_dir /data/LIDC-IDRI \
    --out_dir /data/processed_images
```

| Flag             | Default | Description |
|------------------|---------|-------------|
| `--raw_dir`      | —       | Root of raw DICOM data |
| `--out_dir`      | —       | Output directory for PNG patches |
| `--pad`          | `10`    | Pixel padding around nodule bounding box |
| `--min_ann`      | `2`     | Min radiologist annotations required |
| `--skip_organise`| off     | Skip DICOM folder reorganisation step |

After this step you should see:

```
/data/processed_images/
    Benign_0/      ← class 0 patches
    Malignant_1/   ← class 1 patches
    processed_scans_checkpoint.txt
```

---

### Stage 2 — Training

Trains all four architectures (or a single one via `--model`) with early
stopping, weighted loss, and cosine LR annealing.

```bash
python train.py \
    --data_dir  /data/processed_images \
    --save_dir  /data/checkpoints \
    --epochs    30 \
    --patience  10 \
    --batch_size 32
```

| Flag            | Default        | Description |
|-----------------|----------------|-------------|
| `--data_dir`    | —              | Output of Stage 1 |
| `--save_dir`    | `checkpoints/` | Where to save `.pth` files |
| `--model`       | `all`          | `all` or one of `custom_cnn`, `resnet18`, `efficientnet_b2`, `densenet121` |
| `--epochs`      | `30`           | Max training epochs |
| `--patience`    | `10`           | Early stopping patience (val AUC) |
| `--batch_size`  | `32`           | Batch size |
| `--num_workers` | `2`            | DataLoader workers |

Outputs saved to `--save_dir`:

```
checkpoints/
    custom_cnn_best.pth
    custom_cnn_last.pth
    custom_cnn_history.json
    resnet18_best.pth
    ...
```

---

### Stage 3 — Postprocessing & Evaluation

Evaluates each trained model on the held-out test set and produces plots.

```bash
python postprocessing.py \
    --data_dir     /data/processed_images \
    --ckpt_dir     /data/checkpoints \
    --out_dir      /data/results \
    --gradcam_image /data/processed_images/Malignant_1/LIDC-IDRI-0001_nodule_0_malig_4.5.png
```

| Flag              | Default    | Description |
|-------------------|------------|-------------|
| `--data_dir`      | —          | Same directory used in Stage 2 |
| `--ckpt_dir`      | —          | Directory with `*_best.pth` files |
| `--out_dir`       | `results/` | Where to save plots and JSON metrics |
| `--model`         | `all`      | Evaluate a single model or all |
| `--gradcam_image` | `None`     | Optional nodule PNG for Grad-CAM visualisation |
| `--num_workers`   | `2`        | DataLoader workers |

Outputs:

```
results/
    custom_cnn_metrics.json
    custom_cnn_confusion.png
    resnet18_metrics.json
    ...
    roc_all_models.png
    gradcam_efficientnet_b2_<image_name>.png   (if --gradcam_image used)
```

---

## Models

| Name               | Architecture        | Grad-CAM target layer          |
|--------------------|---------------------|-------------------------------|
| `custom_cnn`       | 4-block CNN + GAP   | `model.block4`                |
| `resnet18`         | ResNet-18           | `model.layer4`                |
| `efficientnet_b2`  | EfficientNet-B2     | `model.features[-1]`          |
| `densenet121`      | DenseNet-121        | `model.features.denseblock4`  |

All heads output a single raw logit; `torch.sigmoid()` is applied at
inference to obtain malignancy probability. Threshold = 0.5 for binary
classification.

---

## Key design choices

| Choice | Rationale |
|--------|-----------|
| BCEWithLogitsLoss + pos_weight | Handles 1.66:1 benign/malignant imbalance |
| WeightedRandomSampler | Balances mini-batches without oversampling entire dataset |
| Skip avg_malignancy == 3.0 | Ambiguous radiologist consensus excluded from labels |
| GAP instead of Flatten | Reduces from ~50k → 256 params before classifier; enables clean Grad-CAM |
| Cosine LR annealing | Smooth decay avoids sharp LR drops that destabilise fine-tuning |
| Early stopping on AUC | AUC more informative than loss for imbalanced medical datasets |

---

## Citation

> Armato III, S.G., et al. "The Lung Image Database Consortium (LIDC) and Image
> Database Resource Initiative (IDRI)." *Medical Physics* 38.2 (2011): 915-931.
