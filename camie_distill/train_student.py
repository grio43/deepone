
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple, List

import deepspeed                           # pip install deepspeed
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MultilabelF1Score

from camie_distill.focal_loss import UnifiedFocalLoss
from camie_distill.student_model import StudentTagger
from camie_distill.flash_stub import install_flash_stub

install_flash_stub()                       # make sure import hooks are in place


# ────────────────────────────────────────────────────────────────────
class CsvDataset(Dataset):
    """Lazy image loader that re‑uses the *exact* preprocessing logic
    used by the teacher (`camie_distill.preprocessing.load_and_preprocess`)."""

    def __init__(
        self,
        csv_path: Path,
        img_dir: Path,
        tag_count: int,
        pad_colour: Tuple[int, int, int],
        fp16: bool,
    ) -> None:
        import pandas as pd
        from camie_distill.preprocessing import load_and_preprocess

        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tag_count = tag_count
        self.pad_colour, self.fp16 = pad_colour, fp16
        self._preprocess = load_and_preprocess
        self._cache: dict[Path, np.ndarray] = {}

    # PyTorch hooks --------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.img_dir / row.file_name

        if path not in self._cache:
            self._cache[path] = self._preprocess(
                path, size=512, pad_colour=self.pad_colour, fp16=self.fp16
            )

        img = torch.from_numpy(self._cache[path])
        tag_idx = list(map(int, str(row.tag_idx).split()))
        target = torch.zeros(self.tag_count, dtype=torch.float32)
        target[tag_idx] = 1.0
        return img, target


# ────────────────────────────────────────────────────────────────────
def _make_deepspeed_config(
    out_path: Path,
    lr: float,
    wd: float,
    train_samples: int,
    mb_size: int,
    accum_steps: int,
) -> None:
    """Write a tiny ZeRO‑2 JSON config to *out_path*."""
    cfg = {
        "train_micro_batch_size_per_gpu": mb_size,
        "gradient_accumulation_steps": accum_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": wd},
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": lr * 1e-2,
                "warmup_max_lr": lr,
                "warmup_num_steps": int(0.25 * train_samples / (mb_size * accum_steps)),
            },
        },
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"},
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
        },
    }
    out_path.write_text(json.dumps(cfg, indent=2))


# ────────────────────────────────────────────────────────────────────
def _evaluate_f1(
    model_engine,
    val_dl: DataLoader,
    thresh: float = 0.5,
) -> tuple[float, float]:
    """Return micro‑F1, macro‑F1 on *val_dl*."""
    model_engine.eval()
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    with torch.no_grad(), torch.cuda.amp.autocast():
        for img, tgt in val_dl:
            img = img.to(model_engine.device, non_blocking=True)
            tgt = tgt.to(model_engine.device, non_blocking=True)

            _, logits_ref = model_engine(img)
            prob = torch.sigmoid(logits_ref).detach().cpu().numpy()
            y_pred.append((prob >= thresh).astype(np.float32))
            y_true.append(tgt.cpu().numpy())

    model_engine.train()
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return micro, macro


# ────────────────────────────────────────────────────────────────────
def main(cfg) -> None:
    # ── metadata ───────────────────────────────────────────────────
    meta = json.loads(Path(cfg.tag_meta).read_text())
    tag_cnt = len(meta["idx_to_tag"])

    # ── model & loss ───────────────────────────────────────────────
    model      = StudentTagger(tag_cnt)
    criterion  = UnifiedFocalLoss()

    metric_micro = MultilabelF1Score(
        num_labels=tag_cnt, average="micro", threshold=0.5
    ).to(model.device if hasattr(model, "device") else "cuda")
    metric_macro = MultilabelF1Score(
        num_labels=tag_cnt, average="macro", threshold=0.5
    ).to(metric_micro.device)

    # ── training dataset / loader ─────────────────────────────────
    ds_train = CsvDataset(
        Path(cfg.csv_path),
        Path(cfg.img_root),
        tag_cnt,
        pad_colour=tuple(cfg.pad_colour),
        fp16=cfg.fp16,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.micro_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ── optional validation loader ────────────────────────────────
    val_dl = None
    if cfg.val_csv:
        ds_val = CsvDataset(
            Path(cfg.val_csv),
            Path(cfg.img_root),
            tag_cnt,
            pad_colour=tuple(cfg.pad_colour),
            fp16=False,
        )
        val_dl = DataLoader(
            ds_val, batch_size=cfg.micro_batch, shuffle=False, num_workers=2
        )

    # ── DeepSpeed engine ──────────────────────────────────────────
    cfg_path = Path(cfg.output_dir) / "ds_config.json"
    _make_deepspeed_config(
        cfg_path,
        cfg.lr,
        cfg.wd,
        len(ds_train),
        cfg.micro_batch,
        cfg.grad_accum,
    )
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=str(cfg_path),
        model_parameters=model.parameters(),
    )

    # ── training loop ─────────────────────────────────────────────
    total_steps = math.ceil(len(dl_train) / cfg.grad_accum) * cfg.epochs
    step = 0
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_file = Path(cfg.output_dir) / "metrics.json"

    for epoch in range(cfg.epochs):
        for img, tgt in dl_train:
            img = img.to(model_engine.device, non_blocking=True)
            tgt = tgt.to(model_engine.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits_i, logits_r = model_engine(img)
                loss = criterion(logits_i, logits_r, tgt)

            model_engine.backward(loss)
            model_engine.step()

            # ── update running F1s ────────────────────────────────
            with torch.no_grad():
                prob = torch.sigmoid(logits_r.detach())
                metric_micro.update(prob, tgt)
                metric_macro.update(prob, tgt)            

            if (
                step % cfg.grad_accum == 0                      # one optimiser step
                and model_engine.global_rank == 0
            ):
                micro_f1 = metric_micro.compute().item()
                macro_f1 = metric_macro.compute().item()

                print(f"[{epoch+1}/{cfg.epochs}] step {step:>6}/{total_steps}  "
                      f"loss={loss.item():.4f}")

            step += 1

        # ── end‑of‑epoch validation ───────────────────────────────
        if model_engine.global_rank == 0 and val_dl is not None:
            micro_f1, macro_f1 = _evaluate_f1(model_engine, val_dl)
            metrics_file.write_text(
                    json.dumps(
                        {
                            "step":  step,
                            "micro_f1": round(micro_f1, 6),
                            "macro_f1": round(macro_f1, 6),
                        },
                        indent=2,
                    )
                )
            metric_micro.reset(), metric_macro.reset()

            print(f"[{epoch+1}/{cfg.epochs}] "
                    f"step {step:>6}  "
                    f"loss={loss.item():.4f}  "
                    f"μ‑F1={micro_f1:.4f}  M‑F1={macro_f1:.4f}")

            step += 1           
            print(
                f"✓ saved metrics – micro‑F1={micro_f1:.4f}, "
                f"macro‑F1={macro_f1:.4f}"
            )


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Student distillation")

    # paths
    p.add_argument("--csv-path", required=True, help="Train CSV produced by dataset_builder")
    p.add_argument("--img-root", required=True, help="Folder containing the images")
    p.add_argument("--tag-meta", required=True, help="metadata.json from the HF model repo")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--val-csv", type=Path, help="CSV with validation ground‑truth")

    # optimisation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)

    # misc
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--pad-colour", type=int, nargs=3, default=(0, 0, 0))
    p.add_argument("--fp16", action="store_true", help="Load images in FP16")

    args = p.parse_args()
    main(args)
