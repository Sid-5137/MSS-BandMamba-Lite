# Usage:
#   python train.py --dataset_root /path/to/musdb18hq --target_stem vocals
#
# In notebook:
#   from bandmamba_light.train import Trainer, TrainConfig
#   trainer = Trainer(TrainConfig(dataset_root="/content/musdb18hq"))
#   trainer.train()

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path

from configs import BandMambaConfig, BASE_CONFIG, SMALL_CONFIG
from model import BandMambaLight, count_parameters
from losses import CombinedLoss
from dataset import create_dataloaders


class TrainConfig:
    """All training hyperparameters in one place."""

    def __init__(
        self,
        # Dataset
        dataset_root: str = "/home/sid/Desktop/Audio_Source_Separation/Datasets/musdb18hq",
        target_stem: str = "vocals",
        chunk_duration: float = 5.0,
        samples_per_track: int = 10,
        # Model
        model_config: BandMambaConfig = BASE_CONFIG,
        # Training
        total_epochs: int = 100,
        warmup_epochs: int = 20,  # Phase 1 (L1) duration
        batch_size: int = 4,
        grad_accumulation: int = 1,  # effective batch = batch_size × grad_accumulation
        num_workers: int = 4,
        # Optimizer
        lr: float = 3e-4,  # conservative LR for stability with Mamba CUDA kernel
        weight_decay: float = 1e-2,
        lr_min: float = 1e-5,  # cosine scheduler minimum
        # Loss
        loss_alpha: float = 1.0,  # weight for time-domain loss
        loss_beta: float = 1.0,  # weight for freq-domain loss
        # Mixed precision
        use_amp: bool = False,  # DISABLED — mamba-ssm CUDA kernel is unstable with fp16
        # Gradient clipping
        max_grad_norm: float = 1.0,  # ← reduced from 5.0 for fp16 stability
        # Checkpointing
        save_dir: str = "./checkpoints",
        save_every: int = 5,  # save checkpoint every N epochs
        log_every: int = 10,  # print loss every N batches
        # Device
        device: str = "auto",  # "auto", "cuda", "cpu"
    ):
        self.dataset_root = dataset_root
        self.target_stem = target_stem
        self.chunk_duration = chunk_duration
        self.samples_per_track = samples_per_track
        self.model_config = model_config
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.grad_accumulation = grad_accumulation
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_min = lr_min
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.save_every = save_every
        self.log_every = log_every

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


class Trainer:
    """
    Handles the full training loop:
      - Two-phase training (L1 warm-up → SI-SDR fine-tune)
      - Mixed precision (fp16) for memory efficiency
      - NaN detection and recovery
      - Gradient accumulation for effective larger batches
      - Cosine learning rate schedule
      - Checkpoint saving and resuming
      - Validation after each epoch
    """

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        print(f"\n{'='*60}")
        print(f"  BandMamba-Light Training Setup")
        print(f"{'='*60}")
        print(f"  Device:          {self.device}")
        print(f"  Target stem:     {config.target_stem}")
        print(
            f"  Epochs:          {config.total_epochs} "
            f"(warmup: {config.warmup_epochs})"
        )
        print(
            f"  Batch size:      {config.batch_size} "
            f"(× {config.grad_accumulation} accumulation "
            f"= {config.batch_size * config.grad_accumulation} effective)"
        )
        print(f"  Learning rate:   {config.lr}")
        print(f"  Grad clip norm:  {config.max_grad_norm}")
        print(f"  Mixed precision: {config.use_amp}")
        print(f"{'='*60}\n")

        # --- Model ---
        self.model = BandMambaLight(config.model_config).to(self.device)
        count_parameters(self.model)

        # --- Loss ---
        self.loss_fn = CombinedLoss(
            phase=1,
            alpha=config.loss_alpha,
            beta=config.loss_beta,
        ).to(self.device)

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # --- LR Scheduler: Cosine annealing ---
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_epochs,
            eta_min=config.lr_min,
        )

        # --- Mixed precision scaler ---
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

        # --- Data ---
        self.train_loader, self.test_loader = create_dataloaders(
            root=config.dataset_root,
            target_stem=config.target_stem,
            chunk_duration=config.chunk_duration,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            samples_per_track=config.samples_per_track,
        )

        # --- Checkpointing ---
        os.makedirs(config.save_dir, exist_ok=True)
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.train_history = []
        self.val_history = []

        # --- NaN tracking ---
        self.nan_count = 0
        self.max_nan_per_epoch = 10  # if more than 10 NaN batches per epoch, stop

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        # Handle DataParallel: save the inner model's state_dict
        model_state = (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "config": vars(self.cfg),
        }

        # Regular checkpoint
        path = os.path.join(self.cfg.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)

        # Best model
        if is_best:
            best_path = os.path.join(self.cfg.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"  ★ New best model saved (val_loss: {val_loss:.4f})")

        # Latest (for easy resume)
        latest_path = os.path.join(self.cfg.save_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle DataParallel: load into inner model
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])

        print(
            f"  Resumed from epoch {self.start_epoch}, "
            f"best_val_loss: {self.best_val_loss:.4f}"
        )

    def train_one_epoch(self, epoch: int) -> dict:
        """Run one training epoch with NaN protection. Returns avg losses."""
        self.model.train()
        total_loss = 0.0
        total_time_loss = 0.0
        total_freq_loss = 0.0
        num_batches = 0
        nan_batches = 0

        self.optimizer.zero_grad()

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            mixture = batch["mixture"].to(self.device, non_blocking=True)  # (B, C, T)
            target = batch["target"].to(self.device, non_blocking=True)  # (B, C, T)

            # Forward pass with mixed precision
            with torch.amp.autocast(self.device.type, enabled=self.cfg.use_amp):
                predicted = self.model(mixture)  # (B, C, T)
                losses = self.loss_fn(predicted, target)
                loss = losses["loss"] / self.cfg.grad_accumulation

            # ===== NaN DETECTION =====
            # If loss is NaN or Inf, skip this batch entirely
            if not math.isfinite(losses["loss"].item()):
                nan_batches += 1
                self.optimizer.zero_grad()  # clear any accumulated gradients
                if nan_batches <= 3:
                    print(
                        f"  ⚠ NaN/Inf loss at batch {batch_idx+1}, skipping (count: {nan_batches})"
                    )
                if nan_batches >= self.max_nan_per_epoch:
                    print(
                        f"  ✗ Too many NaN batches ({nan_batches}), stopping epoch early"
                    )
                    break
                continue

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.cfg.grad_accumulation == 0:
                # Gradient clipping for stability
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.max_grad_norm
                )

                # Skip step if gradients are NaN (scaler handles this automatically,
                # but explicit check adds safety)
                if math.isfinite(grad_norm.item()):
                    self.scaler.step(self.optimizer)
                else:
                    print(f"  ⚠ NaN gradients at batch {batch_idx+1}, skipping update")
                    nan_batches += 1

                self.scaler.update()
                self.optimizer.zero_grad()

            # Track losses
            total_loss += losses["loss"].item()
            total_time_loss += losses["time_loss"].item()
            total_freq_loss += losses["freq_loss"].item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % self.cfg.log_every == 0:
                avg = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                print(
                    f"  [{batch_idx+1}/{len(self.train_loader)}]  "
                    f"loss: {avg:.4f}  lr: {lr:.6f}  "
                    f"time: {elapsed:.1f}s"
                    + (f"  [NaN skips: {nan_batches}]" if nan_batches > 0 else "")
                )

        epoch_time = time.time() - start_time

        if num_batches == 0:
            print("  ✗ No valid batches this epoch!")
            return {
                "loss": float("nan"),
                "time_loss": float("nan"),
                "freq_loss": float("nan"),
                "epoch_time": epoch_time,
            }

        return {
            "loss": total_loss / num_batches,
            "time_loss": total_time_loss / num_batches,
            "freq_loss": total_freq_loss / num_batches,
            "epoch_time": epoch_time,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation with NaN protection. Returns avg losses."""
        self.model.eval()
        total_loss = 0.0
        total_time_loss = 0.0
        total_freq_loss = 0.0
        num_batches = 0

        for batch in self.test_loader:
            mixture = batch["mixture"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.cfg.use_amp
            ):
                predicted = self.model(mixture)
                losses = self.loss_fn(predicted, target)

            # Skip NaN validation batches
            loss_val = losses["loss"].item()
            if not math.isfinite(loss_val):
                continue

            total_loss += loss_val
            total_time_loss += losses["time_loss"].item()
            total_freq_loss += losses["freq_loss"].item()
            num_batches += 1

        if num_batches == 0:
            return {
                "loss": float("inf"),
                "time_loss": float("inf"),
                "freq_loss": float("inf"),
            }

        return {
            "loss": total_loss / num_batches,
            "time_loss": total_time_loss / num_batches,
            "freq_loss": total_freq_loss / num_batches,
        }

    def _recover_from_nan(self):
        """Reload best checkpoint to recover from NaN weights."""
        best_path = os.path.join(self.cfg.save_dir, "best_model.pt")
        latest_path = os.path.join(self.cfg.save_dir, "latest.pt")

        recover_path = best_path if os.path.exists(best_path) else latest_path
        if os.path.exists(recover_path):
            print(f"  ⟳ Recovering from NaN: loading {recover_path}")
            checkpoint = torch.load(
                recover_path, map_location=self.device, weights_only=False
            )
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            # Reduce LR by half after NaN recovery
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * 0.5
            print(f"  ⟳ Reduced LR to {self.optimizer.param_groups[0]['lr']:.6f}")
            return True
        else:
            print("  ✗ No checkpoint found for NaN recovery")
            return False

    def train(self, resume_from: str = None):
        """
        Full training loop with NaN recovery.
        """
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)

        print(f"\nStarting training from epoch {self.start_epoch}...\n")

        nan_recovery_count = 0
        max_nan_recoveries = 5

        for epoch in range(self.start_epoch, self.cfg.total_epochs):
            # --- Phase switch ---
            if epoch == self.cfg.warmup_epochs:
                self.loss_fn.set_phase(2)

            phase = 1 if epoch < self.cfg.warmup_epochs else 2
            phase_str = "L1+STFT" if phase == 1 else "SDR+STFT"

            print(
                f"Epoch {epoch+1}/{self.cfg.total_epochs} "
                f"[Phase {phase}: {phase_str}]"
            )

            # --- Train ---
            train_metrics = self.train_one_epoch(epoch)
            self.train_history.append(train_metrics)

            # --- NaN RECOVERY ---
            if not math.isfinite(train_metrics["loss"]):
                nan_recovery_count += 1
                print(
                    f"  ✗ Epoch produced NaN (recovery attempt {nan_recovery_count}/{max_nan_recoveries})"
                )
                if nan_recovery_count <= max_nan_recoveries:
                    recovered = self._recover_from_nan()
                    if recovered:
                        continue  # retry with recovered weights
                else:
                    print("  ✗ Max NaN recoveries reached. Stopping training.")
                    break

            # --- Validate ---
            val_metrics = self.validate()
            self.val_history.append(val_metrics)

            # --- LR schedule ---
            self.scheduler.step()

            # --- Log ---
            print(
                f"  Train: loss={train_metrics['loss']:.4f}  "
                f"time={train_metrics['time_loss']:.4f}  "
                f"freq={train_metrics['freq_loss']:.4f}  "
                f"({train_metrics['epoch_time']:.1f}s)"
            )
            print(
                f"  Val:   loss={val_metrics['loss']:.4f}  "
                f"time={val_metrics['time_loss']:.4f}  "
                f"freq={val_metrics['freq_loss']:.4f}"
            )

            # --- Checkpointing ---
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            if (epoch + 1) % self.cfg.save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics["loss"], is_best)

            # Reset NaN counter on successful epoch
            nan_recovery_count = 0

            print()

        print("=" * 60)
        print(f"  Training complete!")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoints saved to: {self.cfg.save_dir}")
        print("=" * 60)


# === CLI Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BandMamba-Light")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/sid/Desktop/Audio_Source_Separation/Datasets/musdb18hq",
    )
    parser.add_argument(
        "--target_stem",
        type=str,
        default="vocals",
        choices=["vocals", "drums", "bass", "other"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--chunk_duration", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base",
        choices=["small", "base"],
        help="Model config: 'small' (2.65M) or 'base' (4.18M)",
    )
    args = parser.parse_args()

    # Select model config
    model_config = BASE_CONFIG if args.config == "base" else SMALL_CONFIG

    train_cfg = TrainConfig(
        dataset_root=args.dataset_root,
        target_stem=args.target_stem,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        lr=args.lr,
        chunk_duration=args.chunk_duration,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        save_dir=args.save_dir,
        model_config=model_config,
    )

    trainer = Trainer(train_cfg)
    trainer.train(resume_from=args.resume)
