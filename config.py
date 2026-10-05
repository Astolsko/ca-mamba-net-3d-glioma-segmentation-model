"""Central configuration for the glioma segmentation pipeline.

Edit values here to change an experiment; train.py snapshots this file's
values into every run folder so past runs stay reproducible.
"""
from easydict import EasyDict

cfg = EasyDict()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
cfg.paths = EasyDict()
cfg.paths.root_dir = "/DATA/Abul Hasan/Glioma Revision/data/combined"
cfg.paths.logs_dir = "logs"


cfg.epoch = 60
cfg.learning_rate = 1e-4
cfg.weight_decay = 1e-4
cfg.patience = 50
cfg.grad_clip_norm = 1.0
cfg.val_interval = 1
cfg.val_amp = True
cfg.seed = 0


cfg.warmup_epochs = 5   
cfg.ema_decay = 0.999
cfg.train_oom_skip_limit = 5
cfg.val_oom_skip_limit = 5

# ---------------------------------------------------------------------------
# Checkpointing / crash recovery
# ---------------------------------------------------------------------------
cfg.checkpoint = EasyDict()
cfg.checkpoint.save_last_every_epoch = True
cfg.checkpoint.gc_every_n_val_steps = 20
cfg.checkpoint.cuda_alloc_conf = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Data / dataloaders
# ---------------------------------------------------------------------------
cfg.data = EasyDict()
cfg.data.val_frac = 0.15
cfg.data.test_frac = 0.10
cfg.data.cache_rate = 0.0
cfg.data.batch_size_train = 1
cfg.data.batch_size_val = 1
cfg.data.batch_size_test = 1
cfg.data.num_workers_train = 4
cfg.data.num_workers_val = 4
cfg.data.num_workers_test = 0

cfg.crop = EasyDict()
cfg.crop.fg_threshold = 0
cfg.crop.pos = 2
cfg.crop.neg = 1
cfg.crop.num_samples = 2

# ---------------------------------------------------------------------------
# UNETR model
# ---------------------------------------------------------------------------
cfg.unetr = EasyDict()
cfg.unetr.img_shape = (128, 128, 96)
cfg.unetr.input_dim = 4
cfg.unetr.output_dim = 3
cfg.unetr.patch_size = 16
cfg.unetr.embed_dim = 768
cfg.unetr.num_layers = 12
cfg.unetr.num_heads = 12
cfg.unetr.mlp_dim = 2048
cfg.unetr.extract_layers = [3, 6, 9, 12]
cfg.unetr.dropout = 0.2

# Which encoder feeds the shared decoder (models/unetr.py):
#   "vit"   the original UNETR ViT (blocks/Transformer.py)
#   "mamba" SegMamba-style hierarchical Vision Mamba (blocks/VisionMamba.py)
cfg.unetr.encoder = "mamba"

# ---------------------------------------------------------------------------
# SegMamba encoder
# ---------------------------------------------------------------------------
cfg.mamba = EasyDict()
cfg.mamba.dims = [48, 96, 192, 384]
cfg.mamba.depths = [2, 2, 2, 2]
cfg.mamba.d_state = 16
cfg.mamba.d_conv = 4
cfg.mamba.expand = 2
cfg.mamba.dropout = 0.2

# ---------------------------------------------------------------------------
# ViT encoder (not in use: cfg.unetr.encoder is "mamba")
# ---------------------------------------------------------------------------
# To train with the ViT instead, uncomment the encoder line, or run
# `python train.py --encoder vit`. The code reads the ViT's settings from
# cfg.unetr, so these are the same keys as in the UNETR section above, with
# the values the ViT runs used. Uncomment one only to change it.
# cfg.unetr.encoder = "vit"
# cfg.unetr.patch_size = 16                # 16^3 patches: 8x8x6 = 384 tokens at 128x128x96
# cfg.unetr.embed_dim = 768                # token width (the Mamba path reuses it at 1/16)
# cfg.unetr.num_layers = 12
# cfg.unetr.num_heads = 12
# cfg.unetr.mlp_dim = 2048
# cfg.unetr.extract_layers = [3, 6, 9, 12]  # layers handed to the decoder as skips
# cfg.unetr.dropout = 0.2

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
cfg.loss = EasyDict()
cfg.loss.dice_weight = 0.5
cfg.loss.tversky_weight = 0.3      # weight of the Focal-Tversky term (was focal_weight)
cfg.loss.hausdorff_weight = 0.2    # target weight; annealed 0 -> this over training
cfg.loss.tversky_alpha = 0.7       # FN weight; alpha > beta => recall-focused (helps ET/TC)
cfg.loss.tversky_beta = 0.3        # FP weight
cfg.loss.tversky_gamma = 4.0 / 3.0  # focal exponent is 1/gamma = 0.75 (Abraham & Khan)
cfg.loss.hd_anneal_frac = 0.5      # HD weight reaches full at 50% of epochs, holds after
cfg.loss.aux_z6_weight = 0.3
cfg.loss.aux_z3_weight = 0.15


cfg.metrics = EasyDict()

cfg.metrics.voxel_spacing = (1.0, 1.0, 1.0)
cfg.metrics.auc_every_n_epochs = 10


cfg.infer = EasyDict()


cfg.infer.sw_overlap = 0.75
cfg.infer.sw_mode = "gaussian"
cfg.infer.tta_flips = True
cfg.infer.thresholds = (0.5, 0.5, 0.5)

cfg.infer.tune_thresholds_after_training = True
cfg.infer.min_component_voxels = (0, 0, 176)
cfg.infer.min_total_voxels = (0, 0, 352)


cfg.plot = EasyDict()
cfg.plot.font_family = "serif"
cfg.plot.font_size = 12
cfg.plot.fig_size = (7.0, 4.5)   # inches
cfg.plot.dpi = 150
cfg.plot.formats = ["png"]       # any of png / pdf / svg

cfg.plot.epoch_smoothing = 0.3
cfg.plot.step_smoothing = 0.9


cfg.attention = EasyDict()
cfg.attention.enabled = True
cfg.attention.every_n_epochs = 10
cfg.attention.sample_idx = 0

cfg.visualization = EasyDict()
cfg.visualization.sample_index = 27
cfg.visualization.extra_indices = [0, 5, 10, 15, 20]


cfg.xai = EasyDict()


cfg.xai.run_after_training = True
cfg.xai.components = ["cam", "modality", "uncertainty", "rollout", "faithful"]
cfg.xai.sample_indices = [0, 1, 2]

cfg.xai.cam_layers = [
    "decoder9_upsampler",   # 256ch @ 32x32x24
    "decoder6_upsampler",   # 128ch @ 64x64x48
    "decoder3_upsampler",   #  64ch @ 128x128x96 (full res)
    "decoder0_header.1",    #  64ch, last layer BEFORE the 1x1 output conv
]

cfg.xai.cam_methods = ["hires", "grad"]
cfg.xai.cam_roi = "gt"
cfg.xai.mc_passes = 10
cfg.xai.deletion_fractions = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
cfg.xai.localization_top_frac = 0.05
cfg.xai.peritumoral_radius = 5
