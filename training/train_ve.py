"""
VoiceEncoder Fine-Tuning Script using GE2E Loss
================================================
Fine-tunes the Chatterbox VoiceEncoder on a speaker dataset to improve
speaker identity discrimination.

Usage:
    python training/train_ve.py --config training/train_config.yaml

Or programmatically:
    python training/train_ve.py \
        --data_dir ./my_speaker_data \
        --ve_ckpt path/to/ve.safetensors \
        --output_dir ./checkpoints \
        --num_epochs 30

Data layout:
    data_dir/
        speaker_Alice/
            clip1.wav
            clip2.wav
        speaker_Bob/
            clip1.wav
            ...
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file

# Allow running from project root
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)                          # for `training.*`
sys.path.insert(0, str(Path(_project_root) / "src"))       # for `chatterbox.*`

from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.voice_encoder.config import VoiceEncConfig
from chatterbox.models.voice_encoder.melspec import melspectrogram
from training.data_utils import get_ve_dataloader, VoiceEncoderDataset

import numpy as np


# --------------------------------------------------------------------------- #
#  Wav → Mel helper                                                            #
# --------------------------------------------------------------------------- #

_VE_HP = VoiceEncConfig()

def wav_to_mel(wavs: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of raw waveforms (B, samples) to mel spectrograms
    expected by VoiceEncoder.forward(): (B, T, num_mels).

    Uses the same melspectrogram() function as embeds_from_wavs() so that
    training and inference are consistent.
    """
    mels = []
    for wav in wavs.cpu().numpy():
        mel = melspectrogram(wav, _VE_HP)  # (num_mels, T)
        mel = mel.T                          # → (T, num_mels)
        mels.append(mel)

    # Pad all mels to the same time-length
    max_t = max(m.shape[0] for m in mels)
    padded = np.zeros((len(mels), max_t, _VE_HP.num_mels), dtype=np.float32)
    for i, m in enumerate(mels):
        padded[i, :m.shape[0]] = m

    return torch.from_numpy(padded)


# --------------------------------------------------------------------------- #
#  GE2E Loss (Generalized End-to-End speaker verification loss)               #
# --------------------------------------------------------------------------- #

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End loss for speaker verification.
    Reference: "Generalized End-to-End Loss for Speaker Verification" (Wan et al., 2018)

    Assumes input is shaped (N_speakers, M_clips_per_speaker, embed_dim).
    """

    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds: (N, M, D) — N speakers, M clips each, D-dim L2-normalised embeddings
        Returns:
            Scalar loss.
        """
        N, M, D = embeds.shape
        # L2 normalise
        embeds = embeds / (embeds.norm(dim=-1, keepdim=True) + 1e-8)

        # Centroids for each speaker  (N, D)
        centroids = embeds.mean(dim=1)

        # Similarity matrix  (N*M, N)
        embeds_flat = embeds.view(N * M, D)
        sim = torch.einsum("id,jd->ij", embeds_flat, centroids)  # (N*M, N)
        sim = self.w.abs() * sim + self.b  # scale + shift

        # Ground-truth labels: clip i belongs to speaker (i // M)
        labels = torch.arange(N, device=embeds.device).repeat_interleave(M)
        loss = nn.CrossEntropyLoss()(sim, labels)
        return loss


# --------------------------------------------------------------------------- #
#  Training                                                                     #
# --------------------------------------------------------------------------- #

class VoiceEncoderTrainer:
    """Trainer for VoiceEncoder fine-tuning with GE2E loss."""

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(config["device"])
        self._setup()

    def _setup(self):
        cfg = self.cfg

        # ----- Data -----
        self.loader, self.dataset = get_ve_dataloader(
            data_dir=cfg["data_dir"],
            batch_size=cfg["batch_size"],
            clip_duration=cfg.get("clip_duration", 3.0),
            num_workers=cfg.get("num_workers", 4),
        )

        # ----- Model -----
        self.model = VoiceEncoder()
        if cfg.get("ve_ckpt"):
            ckpt = Path(cfg["ve_ckpt"])
            if ckpt.exists():
                state = load_file(str(ckpt))
                self.model.load_state_dict(state)
                print(f"[VE Train] Loaded checkpoint: {ckpt}")
            else:
                print(f"[VE Train] WARNING: ve_ckpt not found ({ckpt}), training from scratch.")
        self.model.to(self.device)
        self.model.train()

        # ----- Loss -----
        self.ge2e = GE2ELoss().to(self.device)

        # ----- Optimiser -----
        self.optim = optim.Adam(
            list(self.model.parameters()) + list(self.ge2e.parameters()),
            lr=cfg.get("lr", 1e-4),
        )

        # LR scheduler: cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=cfg["num_epochs"] * len(self.loader),
        )

        # AMP (mixed precision)
        self.use_amp = cfg.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Output
        self.output_dir = Path(cfg["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clips_per_speaker = cfg.get("clips_per_speaker", 10)

    def train(self):
        cfg = self.cfg
        best_loss = float("inf")
        M = self.clips_per_speaker  # clips per speaker per GE2E batch

        print(f"\n[VE Train] Starting training for {cfg['num_epochs']} epochs")
        print(f"  Speakers: {self.dataset.num_speakers}")
        print(f"  Device  : {self.device}")
        print(f"  AMP     : {self.use_amp}")
        print(f"  Output  : {self.output_dir}\n")

        for epoch in range(1, cfg["num_epochs"] + 1):
            epoch_loss = 0.0
            t0 = time.time()

            # Collect enough clips per speaker for GE2E
            speaker_clips: dict = {s: [] for s in self.dataset.speakers}

            for wavs, spk_ids in self.loader:
                # Convert raw waveforms → mel spectrograms before passing to model
                mels = wav_to_mel(wavs).to(self.device)  # (B, T, num_mels)
                spk_ids = spk_ids.tolist()

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    embeds = self.model(mels)  # (B, D)

                # Accumulate embeddings per speaker
                for emb, sid in zip(embeds.detach(), spk_ids):
                    speaker_clips[self.dataset.speakers[sid]].append(emb)

                # When every speaker has >= M clips, compute GE2E loss
                if all(len(v) >= M for v in speaker_clips.values()):
                    stacked = torch.stack(
                        [torch.stack(speaker_clips[s][:M]) for s in self.dataset.speakers]
                    )  # (N, M, D)

                    self.optim.zero_grad()
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        loss = self.ge2e(stacked)

                    self.scaler.scale(loss).backward()
                    # Gradient clipping
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.scheduler.step()

                    epoch_loss += loss.item()
                    # Reset accumulators
                    speaker_clips = {s: [] for s in self.dataset.speakers}

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{cfg['num_epochs']} | "
                f"Loss: {epoch_loss:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._save(self.output_dir / "ve_best.safetensors")

            # Periodic checkpoint every N epochs
            save_every = cfg.get("save_every_n_epochs", 5)
            if epoch % save_every == 0:
                self._save(self.output_dir / f"ve_epoch_{epoch:03d}.safetensors")

        # Final save
        self._save(self.output_dir / "ve_final.safetensors")
        print(f"\n[VE Train] Training complete. Best loss: {best_loss:.4f}")
        print(f"[VE Train] Checkpoints saved to: {self.output_dir}")

    def _save(self, fpath: Path):
        save_file(self.model.state_dict(), str(fpath))
        print(f"  → Saved: {fpath.name}")


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Chatterbox VoiceEncoder")
    p.add_argument("--config", type=str, default=None, help="Path to train_config.yaml")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--ve_ckpt", type=str, default=None, help="Path to ve.safetensors")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clips_per_speaker", type=int, default=10)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_yaml_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")


def build_config(args) -> dict:
    cfg = {}
    if args.config:
        cfg = load_yaml_config(args.config)
    # CLI args override yaml
    for key in ["data_dir", "ve_ckpt", "output_dir", "num_epochs",
                "batch_size", "lr", "clips_per_speaker", "use_amp", "device"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg.setdefault(key, val)
    cfg.setdefault("checkpoint_dir", cfg.get("output_dir", "./checkpoints"))
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)

    if not config.get("data_dir"):
        print("ERROR: --data_dir is required (or set data_dir in train_config.yaml)")
        sys.exit(1)

    trainer = VoiceEncoderTrainer(config)
    trainer.train()
