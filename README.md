# Deep ML YOLOv8 project

This project trains and evaluates a YOLOv8 object-detection model. The
examples below use the potholes dataset.

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU is optional. CUDA-enabled PyTorch is used automatically when
  available; otherwise the scripts use the CPU.

## Setup

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell does not allow activation, run scripts with
`.\.venv\Scripts\python.exe` instead of `python`.

## Dataset

Place the potholes dataset under `datasets\potholes`. The training script
expects the split configuration at:

```text
datasets\potholes\new_splits\data.yaml
```

The configuration should point to these folders:

```text
new_splits\
├── train\images and labels\
├── valid\images and labels\
└── test\images and labels\
```

To create the split from an existing `train` and `valid` dataset:

```powershell
.\.venv\Scripts\python.exe -c "from datasets.split_data import split_data; split_data(r'datasets\potholes', r'datasets\potholes\new_splits', already_split=True)"
```

The split uses a fixed seed, so it is reproducible. `new_splits` must not
already exist; the script refuses to overwrite it.

## Train

Train from the YOLOv8-nano architecture without pretrained weights:

```powershell
.\.venv\Scripts\python.exe .\src\train.py --init scratch --data potholes --n_epochs 50
```

To use the epoch count from `configs\train_config.yaml`, omit `--n_epochs`:

```powershell
.\.venv\Scripts\python.exe .\src\train.py --init scratch --data potholes
```

To start from pretrained YOLOv8-nano weights instead:

```powershell
.\.venv\Scripts\python.exe .\src\train.py --init coco --data potholes --n_epochs 50
```

Training output is written below `runs\`. The `best.pt` model weights are
allowed in Git, while other run output and `last.pt` checkpoints are ignored.
The best weights are normally saved at the run's `weights\best.pt` path.

## Evaluate

Evaluate a trained model on the test split:

```powershell
.\.venv\Scripts\python.exe .\src\eval.py `
  --weights_path .\runs\detect\runs\potholes_scratch_50epochs\weights\best.pt `
  --test_data_path .\datasets\potholes\new_splits\data.yaml
```

To force CPU evaluation, add `--device cpu`. To select the first NVIDIA GPU,
use `--device 0`.

## Plot training results

```powershell
.\.venv\Scripts\python.exe .\src\plot_results.py `
  .\runs\detect\runs\potholes_scratch_50epochs\results.csv
```

This creates `results.png` next to the CSV.

## Git and large files

The repository ignores virtual environments, datasets, and most Ultralytics
run output. The trained `best.pt` weights are included because the existing
weights are about 6 MB each, below GitHub's 100 MB per-file limit. The reports
and visualizations from `potholes_scratch_100epochs-4` are also included:
plots, sample images, `results.csv`, and `args.yaml`. If future weights become
large, use Git LFS or external object storage instead.
