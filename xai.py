import argparse
import os

from utils.env_check import ensure_dependencies
ensure_dependencies()

from config import cfg
from utils.dataloader import build_dataloaders
from utils.engine import apply_run_model_config, build_model, get_device, run_inference
from utils import xai


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run folder under logs/")
    parser.add_argument("--only", nargs="+", choices=xai.COMPONENTS,
                        default=list(xai.COMPONENTS),
                        help="Subset of components to run")
    args = parser.parse_args()

    run_dir = os.path.join(cfg.paths.logs_dir, args.run)
    checkpoint_path = os.path.join(run_dir, "checkpoints", "best_metric_model.pth")
    if not os.path.exists(checkpoint_path):
        raise SystemExit(f"No checkpoint at {checkpoint_path}")

    apply_run_model_config(cfg, run_dir)   # the encoder this run was trained with
    device = get_device()
    model = build_model(cfg, device)
    loaders = build_dataloaders(cfg)

    xai.run_xai_suite(
        model=model,
        test_ds=loaders["test_ds"],
        test_loader=loaders["test_loader"],
        checkpoint_path=checkpoint_path,
        out_dir=os.path.join(run_dir, "xai"),
        device=device,
        cfg=cfg,
        # Modality ablation is a relative comparison and every arm gets the same
        # treatment, so TTA would multiply its cost 8x to move no conclusion.
        inferer=lambda m, x: run_inference(m, x, cfg, tta=False),
        components=args.only,
    )


if __name__ == "__main__":
    main()
