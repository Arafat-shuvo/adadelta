# BraTS 2021 Brain Tumor Segmentation: 10 architectures, one fair benchmark (Kaggle)

Ten 2D segmentation architectures trained on **BraTS 2021** with an identical pipeline, optimizer and data splits, so their results can be compared directly. Every notebook is a self-contained, Kaggle-ready PyTorch notebook that trains the model on **3 seeds** and reports **mean ± std**.

| | |
|---|---|
| Task | Multi-label segmentation of **WT** (whole tumor), **TC** (tumor core), **ET** (enhancing tumor) |
| Input | 4 MRI modalities as channels: FLAIR, T1, T1ce, T2 (128×128 slices) |
| Data | BraTS 2021 Task 1 training data (NIfTI), 250-patient pool |
| Optimizer | Adadelta (`lr=1e-3`, `weight_decay=1e-3`) |
| Loss | Soft Dice (no BCE term) |
| Seeds | 42, 43, 44 (different patient split + init + augmentation per seed) |

---

## Notebooks

| Notebook | Model | Notes |
|---|---|---|
| `brats_basic_unet_adadelta_kaggle_250x3seeds.ipynb` | Basic U-Net | Baseline encoder-decoder with plain skip connections |
| `brats_attention_unet_adadelta_kaggle_250x3seeds.ipynb` | Attention U-Net | Additive attention gates on every skip connection |
| `brats_resunet_adadelta_kaggle_250x3seeds.ipynb` | ResUNet | U-Net built from residual blocks |
| `brats_unetpp_adadelta_kaggle_250x3seeds.ipynb` | U-Net++ | Nested, densely connected decoder (plain, no SE) |
| `brats_unet3plus_adadelta_kaggle_250x3seeds.ipynb` | UNet 3+ | Full-scale skip connections, single output head |
| `brats_deeplabv3plus_adadelta_kaggle_250x3seeds.ipynb` | DeepLabV3+ | ASPP on the 1/8 map + low-level decoder, built-in backbone |
| `brats_pspnet_adadelta_kaggle_250x3seeds.ipynb` | PSPNet | Pyramid pooling (bins 1/2/3/6), built-in backbone |
| `brats_fpn_adadelta_kaggle_250x3seeds.ipynb` | FPN | Feature Pyramid Network, built-in backbone |
| `brats_segformer_adadelta_kaggle_250x3seeds.ipynb` | SegFormer | MiT-B0-style transformer encoder + all-MLP decoder |
| `brats_transunet_adadelta_kaggle_250x3seeds.ipynb` | TransUNet | CNN encoder → ViT on the 1/8 bottleneck → CNN decoder |

All models are trained **from scratch** (no pretrained weights) and take 4-channel input.

> **SegFormer:** its embedding dimensions are fixed (32/64/160/256), so `BASE_FILTERS` has no effect on it.
> **FPN:** the pyramid width `fpn_ch=128` is fixed, so `BASE_FILTERS=32` shrinks it by less than the other CNNs.

The notebooks are identical except for the model cell, `MODEL_NAME`, and the title.

---

## Experimental protocol

### Data and preprocessing
- Labels `1/2/4` are converted to three binary masks: **WT** = {1,2,4}, **TC** = {1,4}, **ET** = {4}.
- Each modality is z-scored independently over brain voxels (non-zero), background set to 0.
- Slice selection per patient: all tumor slices, ±2 neighbouring slices, and 10 % of the remaining brain slices; slices with <2 % brain are dropped; at most 100 slices per patient.
- Slices are resized to 128×128 (`INTER_AREA` for images, nearest for masks) and cached on disk as float16 `.npy`.

### Splits (patient-wise, no leakage)
- A fixed **pool of 250 patients** is chosen with `POOL_SEED = 0` and cached once.
- For **each seed**, the pool is re-split by patient: **150 train / 50 val / 50 test** (60/20/20).
- The split depends only on the seed, so **the same seed gives the same split in every notebook**. Models are compared on identical data.

### Training recipe (identical for all models)
- Adadelta, `lr=1e-3`, `weight_decay=1e-3`, `BASE_FILTERS=32`, batch size 32, mixed precision (AMP), up to 200 epochs.
- `ReduceLROnPlateau` on val loss (factor 0.5, patience 4, min lr 1e-7).
- Early stopping on mean val Dice (patience 15); the best-val-Dice checkpoint is used for testing.
- Train-only augmentation, applied identically to image and mask: flips, 90° rotations, scale/zoom (0.85–1.15), gamma, intensity scale/shift.
- `drop_last=True` on the train loader (needed for BatchNorm on 1×1 pooled branches in ASPP/PSP; kept for all models).
- NaN guard and an overall time budget (`TIME_BUDGET_HOURS`).

### Metrics (test set, best checkpoint)
- **Dice, IoU, Precision, Recall**: accumulated over all test slices (micro-averaged), per region and mean.
- **HD95**: 2D per slice, in pixels at 128×128, averaged over slices where both prediction and ground truth are non-empty.
- **Efficiency**: parameters (M), FLOPs (G), inference time (ms, batch 1), peak GPU memory (MB).
  FLOPs come from `thop` (MACs) if installed, otherwise from PyTorch's `FlopCounterMode` (= 2 × MACs). Run every model in the same environment so the numbers are comparable.

---

## Running on Kaggle

1. Create a new notebook and **upload** one of the `.ipynb` files (*File → Import Notebook*).
2. **Add data**: *Add Input* → search for a **BraTS 2021 Task 1** training dataset. Both extracted `*_flair.nii.gz` folders and `.tar` archives are supported (from a tar, only the first 250 patients are unpacked).
3. **Accelerator**: *Settings → Accelerator → GPU* (T4 / P100).
4. Run all cells, or *Save Version → Save & Run All (Commit)*. Results appear in the **Output** tab.

Internet is optional (only for `thop` / `torchinfo`).

### Output layout (`/kaggle/working`)

```
<Model>_Adadelta_seed42/   best_model.pt, training_log.csv, training_curves.png, predictions.png, result.json
<Model>_Adadelta_seed43/   ...
<Model>_Adadelta_seed44/   ...
experiments.csv / .xlsx    one row per seed (all metrics, params, FLOPs, timing, stop reason)
summary_<Model>_Adadelta.csv   mean / std / n over the finished seeds
```

The slice cache lives in `/kaggle/temp`, so it is not saved with the notebook output.

If a session is interrupted, re-run the notebook: seeds with an existing `result.json` are skipped (`FORCE_RERUN = False`).

---

## Configuration

All settings are in the first code cell of each notebook.

| Variable | Default | Meaning |
|---|---|---|
| `NUM_PATIENTS` | 250 | Size of the fixed patient pool (try 300 for more data; slower) |
| `VAL_FRAC`, `TEST_FRAC` | 0.20, 0.20 | Patient-wise split fractions per seed |
| `SEEDS` | `[42, 43, 44]` | One full training run per seed |
| `BASE_FILTERS` | 32 | Width of the CNN models |
| `LR`, `WEIGHT_DECAY` | 1e-3, 1e-3 | Adadelta settings |
| `BATCH_SIZE` | 32 | Lower to 16 on out-of-memory |
| `IMG_SIZE` | 128 | Slice resolution (256 is much slower and larger) |
| `EPOCHS`, `PATIENCE` | 200, 15 | Max epochs, early-stopping patience |
| `TIME_BUDGET_HOURS` | 8.5 | Stops training in time; unfinished seeds are skipped (keep below the Kaggle session limit) |

**For a valid comparison, keep `SEEDS`, `NUM_PATIENTS`, `POOL_SEED`, and every training setting identical across all notebooks.**

### Practical notes
- Three full trainings run back-to-back. Training time varies a lot by model; transformer and nested models (SegFormer, TransUNet, U-Net++) tend to be slower.
- **UNet 3+ and U-Net++** use the most GPU memory. If you must lower the batch size for them, lower it for all models (or report it), otherwise the recipes differ.
- Test sets contain 50 patients per seed, so small Dice differences (~1 %) may fall within seed-to-seed noise. Check the `std` in the summary before ranking models.

---

## Results

Fill in from the `summary_*.csv` files (mean ± std over 3 seeds, test set).

| Model | Mean Dice | WT Dice | TC Dice | ET Dice | Mean IoU | WT HD95 | Params (M) | FLOPs (G) |
|---|---|---|---|---|---|---|---|---|
| Basic U-Net | | | | | | | | |
| Attention U-Net | | | | | | | | |
| ResUNet | | | | | | | | |
| U-Net++ | | | | | | | | |
| UNet 3+ | | | | | | | | |
| DeepLabV3+ | | | | | | | | |
| PSPNet | | | | | | | | |
| FPN | | | | | | | | |
| SegFormer | | | | | | | | |
| TransUNet | | | | | | | | |

---

## Limitations
- 2D slice-wise models at 128×128; no 3D context and no full-resolution evaluation.
- Slice sampling keeps mostly tumor-containing slices, so metrics are not directly comparable to official BraTS challenge scores (which evaluate full 3D volumes).
- A 250-patient subset of BraTS 2021 is used, not the full training set.
- All models are trained from scratch; transformer-based models usually benefit most from pretraining or more data.

---

## References
- Ronneberger et al., *U-Net* (2015)
- Oktay et al., *Attention U-Net* (2018)
- Zhang et al., *Road Extraction by Deep Residual U-Net* (2018)
- Zhou et al., *UNet++* (2018)
- Huang et al., *UNet 3+* (2020)
- Chen et al., *DeepLabV3+* (2018)
- Zhao et al., *PSPNet* (2017)
- Lin et al., *Feature Pyramid Networks* (2017)
- Xie et al., *SegFormer* (2021)
- Chen et al., *TransUNet* (2021)
- Baid et al., *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark* (2021)

## Data and license
The BraTS data is **not** included in this repository. Obtain it through the BraTS challenge / Kaggle and follow its usage terms. Add a license for the code of your choice (e.g. `LICENSE`).
