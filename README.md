# Dual-Encoder RGB + Depth Deepfake Detector

A deepfake (Real vs. Fake) image classifier built on a **symmetric dual EfficientNet-B2** backbone. One encoder processes an RGB–SV fused input (RGB plus Saturation & Value channels from HSV), the other processes a depth map. Features from the two branches are fused **layer-wise by addition** across all 7 stages, and the final stage features are concatenated for classification. A U-Net–style decoder reconstructs the depth map as an auxiliary task that improves feature quality.

## Architecture

```
                 ┌───────────────────────────────┐
  RGB ──► SV ──► │  RGB-SV Fusion (1x1 convs)   │──┐
                 └───────────────────────────────┘  │
                                                    ▼
                                         ┌────────────────────┐
                                         │ EfficientNet-B2    │ ──► stage1..7 feats
                                         │  (RGB-SV branch)   │       │
                                         └────────────────────┘       │
                                                                      │ add
                                         ┌────────────────────┐       │
                                         │ EfficientNet-B2    │ ──► stage1..7 feats
                                         │  (Depth branch)    │
                                         └────────────────────┘
                                                   │
                       ┌───────────────────────────┼───────────────────────────┐
                       ▼                           ▼                           ▼
              Classifier (MLP)           Simplified U-Net Decoder    Fused stage feats (skip)
              1408+1408 → 1               reconstructed depth map
              (Real / Fake)
```

Key design points:

- Dual EfficientNet-B2 encoders, pretrained on ImageNet.
- Stage-wise feature addition (stages 1–7) for fusion.
- Direct skip connections (no channel compression) into the decoder.
- Last-stage feature concatenation (1408 + 1408 = 2816) for classification.
- Multi-task loss: Focal Loss (classification) + MSE (depth reconstruction).

## Repository Layout

```
.
├── main.py          # CLI entry point (argparse)
├── trainer.py       # Train/eval loops + full training pipeline
├── models.py        # Losses, preprocessing blocks, dual encoder, decoder, fusion model
├── dataset.py       # RGB+depth dataset and paired augmentation
├── visualize.py     # Training curves, confusion matrix, t-SNE, reconstruction plots
├── utils.py         # Random seeds, worker/batch-size helpers, HSV conversion, folder checks
├── requirements.txt
└── README.md
```

## Data Layout

Both the RGB and depth roots must mirror each other and contain two class subfolders, `0_real` and `1_fake`:

```
<train-rgb>/                   <train-depth>/
├── 0_real/                    ├── 0_real/
│   ├── img001.png             │   ├── img001.png
│   └── ...                    │   └── ...
└── 1_fake/                    └── 1_fake/
    ├── img101.png                 ├── img101.png
    └── ...                        └── ...
```

The dataset loader matches depth files by the RGB filename stem and tolerates a number of common suffix conventions (`_depth`, `_d`, etc.). Nested class folders are also walked recursively.

## Installation

```bash
git clone https://github.com/<your-user>/DualEncoder-DeepfakeDetection.git
cd DualEncoder-DeepfakeDetection
pip install -r requirements.txt
```

CUDA-enabled PyTorch is recommended; install the appropriate build for your CUDA version from the PyTorch website if the default CPU wheel is not what you want.

## Usage

Validate the data folder structure before training:

```bash
python main.py --check-only \
    --train-rgb   /path/to/train_rgb \
    --train-depth /path/to/train_depth \
    --test-rgb    /path/to/test_rgb \
    --test-depth  /path/to/test_depth \
    --output-dir  /path/to/outputs
```

Train:

```bash
python main.py \
    --train-rgb   /path/to/train_rgb \
    --train-depth /path/to/train_depth \
    --test-rgb    /path/to/test_rgb \
    --test-depth  /path/to/test_depth \
    --output-dir  /path/to/outputs \
    --epochs 25 \
    --lr 1e-4 \
    --cls-weight 1.0 \
    --recon-weight 0.1 \
    --seed 42
```

All CLI options:

| Flag | Default | Description |
|---|---|---|
| `--train-rgb`, `--train-depth` | required | Training RGB / depth root folders |
| `--test-rgb`, `--test-depth` | required | Test RGB / depth root folders |
| `--output-dir` | required | Where checkpoints and visualizations go |
| `--epochs` | `25` | Training epochs |
| `--lr` | `1e-4` | AdamW learning rate |
| `--batch-size` | auto | Override auto-detected batch size |
| `--num-workers` | auto | DataLoader worker count |
| `--seed` | `42` | Random seed |
| `--cls-weight` | `1.0` | Classification loss weight |
| `--recon-weight` | `0.1` | Depth reconstruction loss weight |
| `--check-only` | off | Validate folder structure only, no training |

## Outputs

After training, `--output-dir` contains the best checkpoint and a set of figures:

- `best_dual_encoder_simplified_acc_<acc>.pth` — best checkpoint (by test accuracy)
- `training_metrics_*.png` — loss and accuracy curves
- `confusion_matrix_*.png` — confusion matrix on the test set
- `dual_encoder_reconstruction_sample_*.png` — depth reconstruction examples (4 samples)
- `reconstruction_combined_real_fake.png` — 5 real + 5 fake reconstructions on one figure
- `reconstruction_real_samples.png` / `reconstruction_fake_samples.png` — per-class reconstructions
- `reconstruction_statistics.png` — MSE / MAE distributions (Real vs. Fake)
- `t_sne_plot_*.png` — t-SNE of the 2816-d concatenated features

## Notes

- Input images are resized to 256 and center-cropped to 224×224.
- Training augmentation is paired (RGB and depth get the same geometric transform and the same erasing mask); color jitter is applied to RGB only.
- Batch size and num_workers are auto-tuned from GPU memory / system RAM when not specified; override via CLI if needed.
- Gradient accumulation is used to reach an effective batch size of 32 when the actual batch size is smaller.
