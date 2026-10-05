# Glioma segmentation: overview and run guide

This repo trains and evaluates a 3D network that segments gliomas in
multi-modal brain MRI (FLAIR, T1, T1ce, T2). It predicts three overlapping
regions, each as its own sigmoid channel: tumour core (TC), whole tumour (WT)
and enhancing tumour (ET). The model is a UNETR-style encoder-decoder with a
SegMamba encoder; a ViT encoder can be swapped in with one setting.

This file has two parts. Sections 1-2 explain the code and the architecture.
Sections 3-13 list every runnable entry point, what it writes and roughly what
it costs.

## Contents

1. [Code overview](#1-code-overview)
2. [Architecture](#2-architecture)
3. [Setup](#3-setup)
4. [Quick reference](#4-quick-reference)
5. [Tests](#5-tests)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation-re-score-a-saved-checkpoint)
8. [Explainability (XAI)](#8-explainability-xai)
9. [Switching and comparing encoders](#9-switching-and-comparing-encoders)
10. [ET operating point and paired statistics](#10-et-operating-point-and-paired-statistics)
11. [CSVs and training curves](#11-csvs-and-training-curves)
12. [Data-inspection tools](#12-data-inspection-tools)
13. [Recommended sequence](#13-recommended-sequence)

---

## 1. Code overview

| Path | What it holds |
| --- | --- |
| [config.py](config.py) | Every setting, in one `cfg` object. `train.py` copies it into each run folder as `config_snapshot.json`. |
| [train.py](train.py) | Full pipeline: train, tune thresholds on validation, test, explain. |
| [evaluate.py](evaluate.py) | Re-scores a saved checkpoint with the current inference recipe. No retraining. |
| [xai.py](xai.py) | Runs the explainability suite on a saved checkpoint. |
| [models/unetr.py](models/unetr.py) | The model: `UNETR` (encoder switch, decoder, heads) and `BidirectionalSkip`. |
| [blocks/VisionMamba.py](blocks/VisionMamba.py) | SegMamba encoder: `VisionMambaEncoder`, `GSC`, `MambaLayer`, `TriOrientedMamba`, `SelectiveSSMBranch`, `MlpChannel`. |
| [blocks/Transformer.py](blocks/Transformer.py) | ViT encoder (`Transformer`) and the `DilatedBottleneck`. |
| [blocks/](blocks/) (other files) | Decoder blocks: `Conv3DBlock`, `ConvNeXt3DBlock`, `CoordAtt3D`, `SingleDeconv3DBlock`, `SingleConv3DBlock`, `Deconv3DBlock`. |
| [utils/dataloader.py](utils/dataloader.py) | Dataset discovery, the seeded train/val/test split, z-score normalisation, label conversion. |
| [utils/transforms.py](utils/transforms.py) | Training and validation preprocessing and augmentation. |
| [utils/engine.py](utils/engine.py) | Model construction, the training and validation loops, sliding-window inference, the test pass. |
| [utils/losses.py](utils/losses.py) | Dice + Focal-Tversky + Hausdorff loss and the deep-supervision weighting. |
| [utils/postprocess.py](utils/postprocess.py) | Thresholding, connected-component cleanup, threshold tuning. |
| [utils/metrics.py](utils/metrics.py) | HD95, IoU, sensitivity, specificity, F1, AUC. |
| [utils/xai.py](utils/xai.py), [utils/attention.py](utils/attention.py) | Explainability components and the per-epoch attention overlays. |
| [utils/checkpoint.py](utils/checkpoint.py), [utils/run_logger.py](utils/run_logger.py) | Resumable training state and the run folder layout. |
| [utils/plot.py](utils/plot.py) | Data-check figures and the training curves. |
| [tools/](tools/) | Supervisor script, environment setup, run comparison, operating-point study, inspection tools. |
| [tests/](tests/) | Unit tests on synthetic data. |

---

## 2. Architecture

![Overall architecture](<docs/final arch.drawio.png>)

*Overall network. The left column is the encoder. Each ⊕ is a channel
concatenation.*

The input is a 4-channel 128×128×96 patch at 1 mm spacing. The encoder
produces features at four scales. Each scale passes through a skip encoder,
and the decoder merges them from the deepest scale upward. A full-resolution
branch (the two blocks at the top left) carries the raw input straight to the
output head.

Shapes for a 128×128×96 input, as `channels @ size`:

| Step | Code (`models/unetr.py` unless noted) | Output |
| --- | --- | --- |
| Stem | `Conv3d` 7×7×7, stride 2 | 48 @ 64×64×48 |
| Stage 1 | `GSC` + 2 × `MambaLayer` | 48 @ 64×64×48 (1/2) |
| Stage 2 | downsample + `GSC` + 2 × `MambaLayer` | 96 @ 32×32×24 (1/4) |
| Stage 3 | same | 192 @ 16×16×12 (1/8) |
| Stage 4 | same | 384 @ 8×8×6 (1/16) |
| Skip encoders | `mamba_skip3`, `mamba_skip6`, `mamba_skip9`, `mamba_hidden` | 128 @ 1/2, 256 @ 1/4, 512 @ 1/8, 768 @ 1/16 |
| Upsample + bottleneck | `decoder12_upsampler`, `dilated_bottleneck` | 512 @ 16×16×12 |
| Decoder level 9 | `skip9` + `decoder9_upsampler` | 256 @ 32×32×24 |
| Decoder level 6 | `skip6` + `decoder6_upsampler` | 128 @ 64×64×48 |
| Decoder level 3 | `skip3` + `decoder3_upsampler` | 64 @ 128×128×96 |
| Head | `decoder0` (full-resolution branch) + `decoder0_header` | 3 @ 128×128×96 |

### 2.1 Encoder (SegMamba)

The encoder follows SegMamba (Xing et al., MICCAI 2024) and lives in
[blocks/VisionMamba.py](blocks/VisionMamba.py). Settings are in `cfg.mamba`:
`dims = [48, 96, 192, 384]`, `depths = [2, 2, 2, 2]`, `d_state = 16`,
`d_conv = 4`, `expand = 2`, `dropout = 0.2`.

Each stage runs a gated spatial convolution (GSC) and then two Mamba layers.
Between stages, an instance norm and a 2×2×2 stride-2 convolution halve the
resolution. The Mamba output feeds the next stage. A side branch (instance
norm, then a 1×1×1 MLP) produces the feature that goes to the skip encoder.

![Gated spatial convolution](<docs/GSC.drawio.png>)

*GSC: a 3×3×3 → 3×3×3 branch and a 1×1×1 branch are summed, fused by a 1×1×1
convolution, and added to the input.*

![Mamba layer with the tri-orientated Mamba mixer](<docs/mamba layer tom.drawio.png>)

*Mamba layer (`MambaLayer`). The volume is flattened to tokens and normalised.
The tri-orientated Mamba mixer (`TriOrientedMamba`) scans the tokens in three
orders: raster, reversed, and inter-slice. The three results are mapped back
to raster order and summed. Dropout (p = 0.2) and a residual connection
follow.*

![Selective state-space branch](<docs/ssm_branch_large.drawio.png>)

*One scan branch (`SelectiveSSMBranch`). A causal depthwise convolution
(k = 4) and SiLU give `u`. Linear layers predict the step size Δ and the
matrices B and C from `u`. The selective scan updates a hidden state over the
sequence, and the readout is gated by SiLU(z).*

The dropout is the one deviation from SegMamba, which has none. It is there so
MC-dropout uncertainty (section 8) has layers to sample in the encoder.

### 2.2 Skip encoders and bottleneck

Each encoder scale passes through a MONAI `UnetrBasicBlock` (3×3×3 residual
block with instance norm), which sets the channel count the decoder expects.
The deepest feature (768 channels at 1/16) is upsampled to 1/8 by a transposed
convolution and then enters the dilated bottleneck.

![Dilated bottleneck](<docs/dilated bottleneck.drawio.png>)

*Dilated bottleneck (`DilatedBottleneck`): three parallel 3×3×3 convolutions
with dilation 1, 2 and 4 are concatenated, fused by a 1×1×1 convolution with
GroupNorm and ReLU, and added to the input.*

### 2.3 Bidirectional skip connection

At each decoder level, the skip feature from the encoder (shallow) and the
feature coming up from the level below (deep) are fused before decoding.

![Bidirectional skip connection](<docs/bsc.drawio.png>)

*BSC (`BidirectionalSkip`): each input is scaled by a learned scalar passed
through a sigmoid. The two are concatenated and reduced by a 1×1×1
convolution, GroupNorm and ReLU. The decoder then concatenates the fused
feature with the deep feature again.*

### 2.4 Decoder stage

Every decoder level applies the same four blocks in order: a feature
refinement block (`Conv3DBlock`: 3×3×3 convolution, GroupNorm, ReLU), a
ConvNeXt-V2 block, 3D coordinate attention, and a 2×2×2 transposed convolution
that doubles the resolution.

![ConvNeXt 3D block](<docs/ConvNeXt3DBlock_diagram (2).drawio (1).png>)

*ConvNeXt block (`ConvNeXt3DBlock`): 7×7×7 depthwise convolution, layer norm,
1×1×1 expansion to 4C, GELU, global response normalisation, 1×1×1 projection
back to C, layer scale, residual.*

![3D coordinate attention](<docs/coordatt3d.drawio.png>)

*Coordinate attention (`CoordAtt3D`): the feature map is average-pooled along
each of the three axes. The pooled vectors share one 1×1×1 convolution, are
split back per axis, and become three sigmoid gates that multiply the input.
Dropout (p = 0.1) follows.*

### 2.5 Output head and deep supervision

The full-resolution branch (`decoder0`, two feature refinement blocks on the
raw input) is concatenated with the last decoder output. Two more refinement
blocks and a 1×1×1 convolution give the three output channels.

During training the model also returns two auxiliary predictions, taken from
the outputs of decoder levels 6 and 3 (`aux_head_z6`, `aux_head_z3`). They are
upsampled to the patch size and add to the loss with weights 0.3 and 0.15. At
inference only the main output is returned.

![3D view of a segmented case](<docs/3d results view.png>)

*Three views of one case in 3D: edema in red, enhancing tumour in yellow,
necrotic and non-enhancing core in blue.*

### 2.6 The ViT encoder option

With `cfg.unetr.encoder = "vit"` the encoder is UNETR ViT: 12
transformer layers over 16×16×16 patches (384 tokens of width 768). Layers 3,
6, 9 and 12 are reshaped to the 8×8×6 grid and upsampled by transposed
convolutions to the same four skip shapes the Mamba path produces. Everything
from the bottleneck onward is the same module graph, so a ViT run and a Mamba
run differ in the encoder only. The ViT settings are the `cfg.unetr` keys;
config.py lists them in a commented block.

### 2.7 Data, loss and inference

**Data.** `cfg.paths.root_dir` holds one folder per patient, named `<id>`,
with `<id>_flair.nii`, `<id>_t1.nii`, `<id>_t1ce.nii`, `<id>_t2.nii` and
`<id>_seg.nii`. Patients are shuffled with `cfg.seed` and split into test
(`cfg.data.test_frac`, 10%), validation (`cfg.data.val_frac`, 15%) and
training (the rest). Labels become three channels: TC = labels 1 and 4, WT =
labels 1, 2 and 4, ET = label 4.

**Preprocessing.** Every split is reoriented to RAS, resampled to 1 mm,
cropped to the brain's bounding box, and z-scored per modality over the brain
voxels. Training then draws `cfg.crop.num_samples` (2) patches of 128×128×96
per volume, centred on tumour with probability pos / (pos + neg) = 2/3, and
applies flips, a small affine, and intensity augmentations. Validation and
test keep the whole cropped brain.

**Loss.** 0.5 × Dice + 0.3 × Focal-Tversky (α = 0.7, β = 0.3, so false
negatives cost more) + 0.2 × Hausdorff distance-transform loss. The Hausdorff
weight ramps from 0 to full over the first half of training. The auxiliary
heads use Dice + Focal-Tversky only.

**Optimisation.** AdamW (lr 1e-4, weight decay 1e-4), 5 warmup epochs then
cosine decay to 1e-6, mixed precision, gradient clipping at 1.0, and an
exponential moving average of the weights (decay 0.999). Validation and the
saved best checkpoint use the averaged weights. The best checkpoint is the
epoch with the highest validation mean Dice.

**Inference.** Sliding window with a 128×128×96 window, Gaussian blending and
`cfg.infer.sw_overlap` (0.75). The test pass averages over the 8 axis-flip
combinations (test-time augmentation, TTA), applies per-channel thresholds
tuned on validation, and removes small connected components for ET
(`cfg.infer.min_component_voxels`, `cfg.infer.min_total_voxels`).

---

## 3. Setup

All commands run from the repo root.

1. **Environment.** [requirements.txt](requirements.txt) is the full
   package list of the `pytorch2` conda env (torch 2.11, MONAI 1.3). It runs
   everything with the ViT encoder.
2. **Mamba env.** The Mamba encoder needs the `mamba_ssm` CUDA kernels, which
   do not load in `pytorch2`. `tools/setup_mamba_env.sh` builds a `mamba` env:
   a copy of `pytorch2` with torch 2.6, mamba_ssm 2.2.4 and causal_conv1d
   1.5.0.post8. The conda and env paths are set at the top of that script, so
   edit them for another machine. The `mamba` env runs every command in this
   file. `train.py` refuses to start a Mamba run on a GPU without the kernels.
3. **Paths.** Set `cfg.paths.root_dir` in [config.py](config.py) to the
   dataset folder (layout in section 2.7). Run folders go under
   `cfg.paths.logs_dir` (default `logs/`).

Timings in this file were measured on one RTX A6000 (48 GB). `train.py` prints
the parameter count of the model it built at startup.

---

## 4. Quick reference

| Command | Retrains? | Cost | Writes to |
| --- | --- | --- | --- |
| `python -m pytest tests` | no | minutes | nothing |
| `python train.py --name <run>` | **yes** | ~45 h (60 epochs) | `logs/<run>/` (everything) |
| `python train.py --name <run> --resume` | **yes** | remaining epochs | `logs/<run>/` (same folder) |
| `tools/train_supervisor.sh <run>` | **yes** | training + restarts | `logs/<run>/` (same folder) |
| `python evaluate.py --run <run> --tune-thresholds` | no | ~2 h | `logs/<run>/eval/` |
| `python xai.py --run <run>` | no | ~30-45 min | `logs/<run>/xai/` |
| `python tools/compare_runs.py <a> <b>` | no | seconds | `logs/compare_<a>_vs_<b>/` |
| `python tools/cache_predictions.py --run <run> --out <dir>` | no | ~1-1.5 h per run (GPU) | `<dir>/<run>/` (~8 GB) |
| `python tools/operating_point_study.py --cache <dir> --runs <a> <b>` | no | ~20-30 min (CPU) | `logs/compare_<a>_vs_<b>/` |
| `tools/setup_mamba_env.sh` | no | ~10 min | the `mamba` conda env |
| `python tools/crop_visual_check.py` | no | seconds | PNGs beside the script |
| `python tools/nifti_viewer.py` | no | seconds | interactive window |

---

## 5. Tests

```bash
python -m pytest tests                       # whole suite
python -m pytest tests -m "not gpu"          # skip the CUDA-only tests
python -m pytest tests/test_xai.py -v        # one file
python -m pytest tests -k threshold          # one topic
```

The tests use synthetic data only. They catch config and wiring mistakes that
would otherwise surface hours into a training run, so run them before any long
job.

---

## 6. Training

```bash
conda activate mamba
python train.py --name <run>
python train.py --name <run> --encoder vit     # one launch with the ViT encoder
```

Without `--name` the script prompts for one. It runs these stages in order:

1. **Train** for `cfg.epoch` epochs (60), validating every `cfg.val_interval`
   epochs. Training stops early after `cfg.patience` epochs without a better
   validation mean Dice.
2. **Tune thresholds** on the validation split
   (`cfg.infer.tune_thresholds_after_training`).
3. **Test** with the full inference recipe: tuned thresholds, TTA,
   post-processing.
4. **XAI suite** (`cfg.xai.run_after_training`), all five components.

Set either flag to `False` in config.py to skip stage 2 or 4. A failing XAI
component cannot discard the run: the error is recorded in `xai/summary.json`
and the checkpoint is reloaded.

Per-epoch validation runs without TTA. It uses `cfg.infer.val_sw_overlap` when
that key is set and `cfg.infer.sw_overlap` (0.75) when it is not. config.py
does not set it at the moment.

Run folder:

```text
logs/<run>/
  log.txt                full stdout and stderr
  config_snapshot.json   cfg as it was at run start
  metrics.csv            one row per epoch
  train_steps.csv        one row per optimizer step
  val_steps.csv          one row per validation patient per epoch
  checkpoints/           best_metric_model.pth (weights), last.pth (full state)
  plots/                 data split + loss, dice, iou, hd95, sens, f1, lr, loss_steps
  visualizations/        sample modalities and labels, drawn before training
  attention/             attention overlays saved during validation
  eval/                  threshold_sweep.json, infer_config.json
  testing/               test_metrics.csv + qualitative outputs
  xai/                   figures, one JSON per component, summary.json
```

**Attention overlays.** With `cfg.attention.enabled = True`, every
`cfg.attention.every_n_epochs` epochs the validation loop saves an attention
overlay for validation sample `cfg.attention.sample_idx` into `attention/`.

### 6.1 Resuming an interrupted run

Every epoch writes `checkpoints/last.pth`: weights, the averaged weights, the
optimizer, the LR scheduler, the gradient scaler, the random-number state and
the best-metric bookkeeping. A crash costs at most one epoch.

```bash
python train.py --name <run> --resume          # continue; fails if there is no last.pth
python train.py --name <run> --auto-resume     # continue if possible, else start fresh
python train.py --name <run> --resume-from logs/other/checkpoints/last.pth
```

A resumed run writes into the same `logs/<run>/` folder and appends to the
existing CSVs. Rows at or past the resume epoch are dropped first, so a
replayed epoch never appears twice.

`--resume` refuses to continue if `cfg.unetr.img_shape` or the encoder changed
since the checkpoint was written. Changing `cfg.epoch` is allowed but warns:
the LR schedule was built for the old budget and is restored as it was.

### 6.2 Unattended runs

```bash
nohup tools/train_supervisor.sh <run> > logs/<run>-supervisor.log 2>&1 &
tail -f logs/<run>/log.txt
```

The supervisor relaunches `train.py --name <run> --auto-resume` on any
non-zero exit, up to 20 times (`tools/train_supervisor.sh <run> 40` for
another limit). It waits `BACKOFF_SECONDS` (default 60) between attempts so
the dead process releases its GPU memory. A clean finish or Ctrl-C ends the
loop. It uses the active env's `python`; set `PYTHON_BIN` to use another.
Arguments after `--` are passed to `train.py`.

There are two recovery layers:

| Layer | Handles | Cost |
| --- | --- | --- |
| In-process skip (`utils/engine.py`) | one batch or validation volume that runs out of memory | that sample is dropped from the epoch |
| Supervisor restart | repeated out-of-memory errors, a killed process, a dead CUDA context | replay of at most one epoch |

`cfg.train_oom_skip_limit` and `cfg.val_oom_skip_limit` (5 each) bound the
first layer. Exceeding either raises, which hands over to the second.

### 6.3 GPU memory

Training allocates a fixed patch shape every step. Validation feeds a whole
cropped brain, which has a different shape for every patient. Alternating the
two fragments the allocator's reserved memory, and an earlier run died at
epoch 33 with most of the card nominally free. These measures address it:

1. `train.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before
   CUDA starts (`cfg.checkpoint.cuda_alloc_conf`). Override it from the shell
   if needed.
2. The cache is emptied at every switch between training and validation.
3. Attention is captured only on the epochs that save an overlay, and to CPU.
4. The CPU copy of the logits for AUC is made only every
   `cfg.metrics.auc_every_n_epochs` epochs.
5. Per-step tensors are released before the next batch is loaded.

Every epoch line in `log.txt` ends with a memory reading:

```text
gpu_mem alloc=3.4G reserved=23.3G peak=21.4G retries=0 ooms=0
```

`reserved` climbing while `alloc` stays flat is fragmentation. A non-zero
`retries` means the allocator is already flushing to satisfy requests, which
is the last warning before an out-of-memory error. A crash leaves its
traceback in `log.txt`.

If memory still runs out, these settings shrink the working set, in increasing
order of how much they change the experiment: `cfg.crop.num_samples` (2 → 1),
a lower sliding-window overlap, and `cfg.unetr.img_shape`. A run cannot be
resumed across a change to the last one.

---

## 7. Evaluation: re-score a saved checkpoint

```bash
python evaluate.py --run <run>                              # current recipe as-is
python evaluate.py --run <run> --tune-thresholds            # tune on validation, then test
python evaluate.py --run <run> --tag notta --no-tta         # without TTA
python evaluate.py --run <run> --tag raw --no-postprocess   # without component cleanup
```

Outputs go to `logs/<run>/eval/`: `test_metrics_<tag>.csv`,
`infer_config_<tag>.json`, `threshold_sweep_<tag>.json` when tuning, and
`visualizations/`. The default tag is `eval`. The run's own `testing/` folder
is left untouched, and a distinct `--tag` keeps one ablation from overwriting
another.

Everything this script changes happens at inference time: overlap and
blending, TTA, thresholds, component cleanup (all in `cfg.infer`). The model
is rebuilt with the encoder recorded in the run's `config_snapshot.json`, so
config.py does not need to match the run.

Thresholds are tuned on validation and reported on test. Tuning on test would
be tuning on the number being reported.

---

## 8. Explainability (XAI)

```bash
python xai.py --run <run>                          # all five components
python xai.py --run <run> --only cam               # one component
python xai.py --run <run> --only modality faithful
```

| Component | What it produces |
| --- | --- |
| `cam` | Seg-Grad-CAM and HiResCAM overlays at each layer in `cfg.xai.cam_layers`, for each of the three classes |
| `modality` | A heatmap of the Dice lost when each MRI sequence is removed |
| `uncertainty` | MC-dropout entropy maps and error-retention curves |
| `rollout` | ViT run: attention rollout through the 12 blocks. Mamba run: hidden-attention rollout of the deepest stage, on the same 8×8×6 grid |
| `faithful` | Deletion curves against a random baseline, localisation scores, and a weight-randomisation sanity check |

Results go to `logs/<run>/xai/`: figures, `<component>.json` and
`summary.json`. The checkpoint on disk is never modified.

`faithful` always runs last, whatever order is passed. It randomises the model
weights, so anything after it would explain a destroyed model. The checkpoint
is reloaded straight afterwards.

Settings are in `cfg.xai`:

| Key | Meaning |
| --- | --- |
| `components` | Which components `train.py` runs at the end of training |
| `sample_indices` | Test samples to explain. Each extra sample multiplies the cost of `faithful`. |
| `cam_layers`, `cam_methods` | Decoder layers and CAM variants (`hires`, `grad`) |
| `cam_roi` | Region the CAM score is summed over: `gt`, `pred` or `all` |
| `mc_passes` | Stochastic forward passes for MC-dropout |
| `deletion_fractions` | Fractions of top-attributed voxels deleted in the faithfulness sweep |
| `localization_top_frac`, `peritumoral_radius` | Settings of the localisation score |

Two things to know when reading the results:

- Read `xai/faithful.json` first. If the randomisation similarity does not
  decay, or the CAM deletion curve is no better than the random baseline, the
  maps carry no information.
- The CAM at `decoder0_header.1` sits one 1×1×1 convolution from the output,
  so it largely redraws the prediction. Use the deeper decoder layers to judge
  localisation.

---

## 9. Switching and comparing encoders

```bash
python train.py --name <run> --encoder mamba   # or: cfg.unetr.encoder in config.py
python train.py --name <run> --encoder vit
python tools/compare_runs.py <run_a> <run_b>
python tools/compare_runs.py <run_a> <run_b> --matched-epoch 30
```

`--encoder` overrides `cfg.unetr.encoder` for one launch and is recorded in
the run's `config_snapshot.json`. `evaluate.py` and `xai.py` read the encoder
from that snapshot.

`compare_runs.py` writes `logs/compare_<a>_vs_<b>/`: `curves.png` (validation
Dice, ET HD95 and training loss per epoch, both runs on one axis),
`summary.md` and `summary.csv` (parameters, minutes per epoch, validation at
the matched epoch and at each run's best, test metrics, XAI numbers).

Measured cost of the two encoders on this workstation: the Mamba run took
44.8 min per epoch with a 22.6 GB peak, the ViT run 41.8 min per epoch.

When comparing two runs, compare them at a matched epoch as well as at their
best. Runs with different epoch budgets have different cosine schedules. The
two envs also use different torch builds (2.11 and 2.6).

---

## 10. ET operating point and paired statistics

```bash
C=/some/scratch/pred_cache            # ~8 GB per run; keep it outside logs/
python tools/cache_predictions.py --run <run_a> --out $C
python tools/cache_predictions.py --run <run_b> --out $C
python tools/operating_point_study.py --cache $C --runs <run_a> <run_b>
```

`cache_predictions.py` runs a run's checkpoint with its own
`eval/infer_config.json` over validation and test, once, and saves the sigmoid
probabilities. On test it also runs the modality ablation for every patient.
Run each run in the env it was trained in, so the numbers match its logged
test pass. `--limit 2` gives a quick check, `--no-modality` skips the
ablation, and a re-run skips patients already cached.

`operating_point_study.py` needs no GPU. Its report has five parts:

1. **Anchor check.** The cache must reproduce `testing/test_metrics.csv` at
   the shipped thresholds. If it does not, do not read further.
2. **ET threshold chosen on validation.** The rule is fixed in the script: the
   lowest ET threshold that reaches validation's minimum count of patients
   with a false or missed ET region. Test is shown at that point.
3. **Paired test comparison** of the two runs on the same patients, with
   bootstrap confidence intervals and Wilcoxon p-values.
4. **Patients with an empty prediction or empty ground truth**, side by side.
5. **Modality ablation over the whole test split**, with confidence intervals.

Outputs: `logs/compare_<a>_vs_<b>/operating_point_study.md` and `.json`, and
`logs/<run>/eval/per_patient_test.csv`.

---

## 11. CSVs and training curves

`train.py` draws `plots/` from the CSVs at the end of training, with the style
in `cfg.plot` (font, size, dpi, formats, smoothing). Every value in the CSVs
is raw.

| File | One row per | Columns |
| --- | --- | --- |
| `metrics.csv` | epoch | lr, train and validation loss, per-region and mean Dice, HD95, sensitivity, specificity, IoU, F1, AUC |
| `train_steps.csv` | optimizer step | epoch, step, lr, step time, total loss, main-head loss, both auxiliary losses, the three unweighted loss terms, Hausdorff ramp |
| `val_steps.csv` | validation patient, per epoch | epoch, patient, step time, loss and its terms, every metric per region |

Points that are easy to misread:

- `train_loss` in `metrics.csv` includes the auxiliary heads; `val_loss` is
  the main head only. Compare `val_loss` with `loss_main`, the dashed line in
  `loss.png`.
- Region Dice in `metrics.csv` averages over patients whose ground truth has
  that region. In `val_steps.csv` such a patient's `dice_*` is NaN.
- AUC is computed every `cfg.metrics.auc_every_n_epochs` epochs. On other
  epochs `val_steps.csv` leaves it blank and `metrics.csv` writes 0.
- Smoothing (`cfg.plot.epoch_smoothing`, `cfg.plot.step_smoothing`) is applied
  only when drawing. The raw curve stays visible under it. Take reported
  numbers from the CSVs, never from a smoothed curve.
- `lr` in `metrics.csv` is logged after that epoch's scheduler step, so it is
  the next epoch's rate. `lr` in `train_steps.csv` is the rate each step used.

---

## 12. Data-inspection tools

These are manual tools, not part of the pipeline.

```bash
python tools/nifti_viewer.py         # file dialogs pick the image and segmentation
python tools/crop_visual_check.py    # set SAMPLE_DIR, START_SLICE, THRESHOLD in the script first
```

`crop_visual_check.py` exercises the older fixed-depth-window crop
(`CropRawDepthd` and `CropForegroundHWd` in `utils/transforms.py`). The
current training pipeline crops to the brain's bounding box instead and does
not use those classes.

---

## 13. Recommended sequence

```bash
conda activate mamba
python -m pytest tests                                  # 1. wiring is sound
nohup tools/train_supervisor.sh <run> > logs/<run>-supervisor.log 2>&1 &
                                                        # 2. train, tune, test, explain
python evaluate.py --run <run> --tag notta --no-tta     # 3. inference ablations
python evaluate.py --run <run> --tag raw --no-postprocess
python xai.py --run <run> --only cam                    # 4. redo one XAI component
python tools/compare_runs.py <baseline> <run>           # 5. compare against another run
```

---
