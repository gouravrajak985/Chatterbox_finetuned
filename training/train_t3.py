"""
T3 Fine-Tuning with LoRA (Low-Rank Adaptation)
================================================
Fine-tunes the T3 (Llama 520M) model for a specific speaker using LoRA,
which is memory-efficient and preserves the original model's capabilities.

Requirements:
    pip install peft

Usage:
    python training/train_t3.py --config training/train_config.yaml

Or:
    python training/train_t3.py \\
        --data_dir ./my_speaker_data \\
        --model_dir ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/... \\
        --output_dir ./t3_lora_output \\
        --num_epochs 10

Data layout (paired text + audio):
    data_dir/
        speaker_Alice/
            utterance001.wav
            utterance001.txt   ← transcript of utterance001.wav
            utterance002.wav
            utterance002.txt
            ...

Notes:
    - LoRA is applied ONLY to the LlamaModel attention layers (q_proj, v_proj, k_proj, o_proj)
    - T3 text/speech embeddings and projection heads are also fine-tuned (small but impactful)
    - The base T3 weights remain frozen; only LoRA deltas + heads are updated
    - After training, merge LoRA weights and save a full t3_cfg.safetensors
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

# Allow running from project root
_project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _project_root)                          # for `training.*`
sys.path.insert(0, str(Path(_project_root) / "src"))       # for `chatterbox.*`

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR, drop_invalid_tokens
from training.data_utils import get_t3_dataloader, load_and_resample, S3GEN_SR


# --------------------------------------------------------------------------- #
#  LoRA setup                                                                   #
# --------------------------------------------------------------------------- #

def apply_lora_to_t3(t3: T3, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """
    Apply LoRA to T3's internal LlamaModel attention projections.

    Args:
        t3:          T3 instance.
        lora_r:      LoRA rank.
        lora_alpha:  LoRA alpha (scaling = alpha / r).
        lora_dropout: Dropout on LoRA layers.

    Returns:
        t3 with LoRA applied, peft PeftModel wrapping tfmr.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "PEFT library required for T3 LoRA fine-tuning.\n"
            "Install with: pip install peft"
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Llama attention projections
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Wrap the inner transformer (LlamaModel)
    t3.tfmr = get_peft_model(t3.tfmr, lora_config)
    t3.tfmr.print_trainable_parameters()

    # Also unfreeze text/speech embeddings and projection heads (lightweight but important)
    for param in t3.text_emb.parameters():
        param.requires_grad = True
    for param in t3.speech_emb.parameters():
        param.requires_grad = True
    for param in t3.speech_head.parameters():
        param.requires_grad = True
    for param in t3.cond_enc.parameters():
        param.requires_grad = True

    return t3


def save_lora_model(t3: T3, output_dir: Path, merge: bool = True):
    """
    Save T3 LoRA weights.

    If merge=True:  merges LoRA into base weights and saves a complete t3_cfg.safetensors
    If merge=False: saves only the LoRA adapter weights (smaller, but needs PEFT at inference)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if merge:
        print("[T3 Train] Merging LoRA weights into base model...")
        try:
            t3.tfmr = t3.tfmr.merge_and_unload()
        except AttributeError:
            pass  # Already merged or not peft model

        state_dict = t3.state_dict()
        save_file(state_dict, str(output_dir / "t3_cfg.safetensors"))
        print(f"[T3 Train] Merged model saved → {output_dir / 't3_cfg.safetensors'}")
    else:
        t3.tfmr.save_pretrained(str(output_dir / "lora_adapter"))
        # Save non-LoRA trainable params separately
        extra_state = {
            "text_emb": t3.text_emb.state_dict(),
            "speech_emb": t3.speech_emb.state_dict(),
            "speech_head": t3.speech_head.state_dict(),
            "cond_enc": t3.cond_enc.state_dict(),
        }
        torch.save(extra_state, output_dir / "t3_extra_params.pt")
        print(f"[T3 Train] LoRA adapter saved → {output_dir / 'lora_adapter'}")


# --------------------------------------------------------------------------- #
#  T3 Collation with S3 tokenization                                           #
# --------------------------------------------------------------------------- #

def collate_t3_batch(batch_dict: dict, model: ChatterboxTTS, device: torch.device):
    """
    Convert a raw batch (texts + audio paths) into T3 training tensors.

    Returns:
        t3_cond:           T3Cond conditioner (speaker info)
        text_tokens:       (B, L_text) padded
        text_token_lens:   (B,)
        speech_tokens:     (B, L_speech) padded
        speech_token_lens: (B,)
    """
    texts = batch_dict["texts"]
    audio_paths = batch_dict["audio_paths"]
    B = len(texts)

    from chatterbox.tts import punc_norm

    # ----- Text tokens -----
    all_text_tokens = []
    for text in texts:
        text = punc_norm(text)
        tok = model.tokenizer.text_to_tokens(text).squeeze(0).cpu()  # (L,) — keep on CPU for now
        # Add SOT/EOT on same device as tok
        sot = model.t3.hp.start_text_token
        eot = model.t3.hp.stop_text_token
        tok = torch.cat([tok.new_tensor([sot]), tok, tok.new_tensor([eot])])
        all_text_tokens.append(tok)

    text_lens = torch.tensor([t.size(0) for t in all_text_tokens])
    max_tl = text_lens.max().item()
    text_tokens_padded = torch.zeros(B, max_tl, dtype=torch.long)
    for i, tok in enumerate(all_text_tokens):
        text_tokens_padded[i, :tok.size(0)] = tok

    # ----- Speech tokens (via S3Tokenizer) -----
    all_speech_tokens = []
    all_conds = []
    for ap in audio_paths:
        model.prepare_conditionals(ap)
        cond = model.conds
        all_conds.append(cond)

        # Get speech tokens from the audio via S3Tokenizer
        wav_16k = load_and_resample(ap, S3_SR)
        wav_16k_t = torch.from_numpy(wav_16k).unsqueeze(0)
        s3_tok = model.s3gen.tokenizer
        tokens, _ = s3_tok.forward([wav_16k], max_len=1000)
        tokens = drop_invalid_tokens(tokens.squeeze(0)).cpu()  # keep on CPU for now
        tokens = tokens[tokens < 6561]

        # Add SOS/EOS on same device as tokens
        sos = model.t3.hp.start_speech_token
        eos = model.t3.hp.stop_speech_token
        tokens = torch.cat([tokens.new_tensor([sos]), tokens, tokens.new_tensor([eos])])
        all_speech_tokens.append(tokens)

    speech_lens = torch.tensor([t.size(0) for t in all_speech_tokens])
    max_sl = speech_lens.max().item()
    speech_tokens_padded = torch.zeros(B, max_sl, dtype=torch.long)
    for i, tok in enumerate(all_speech_tokens):
        speech_tokens_padded[i, :tok.size(0)] = tok

    # ----- Average T3Cond across batch (use first speaker's cond for now) -----
    # For single speaker fine-tuning, all conds should be the same speaker anyway
    t3_cond = all_conds[0].t3.to(device=device)

    return (
        t3_cond,
        text_tokens_padded.to(device),
        text_lens.to(device),
        speech_tokens_padded.to(device),
        speech_lens.to(device),
    )


# --------------------------------------------------------------------------- #
#  Trainer                                                                      #
# --------------------------------------------------------------------------- #

class T3LoRATrainer:
    """Trainer for T3 LoRA fine-tuning."""

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device(config["device"])
        self._setup()

    def _setup(self):
        cfg = self.cfg

        # ----- Data -----
        self.loader, self.dataset = get_t3_dataloader(
            data_dir=cfg["data_dir"],
            batch_size=cfg.get("t3_batch_size", 2),
            num_workers=cfg.get("num_workers", 0),  # 0 recommended for debugging
        )

        # ----- Load full Chatterbox model (for speech tokenisation & conditioning) -----
        model_dir = cfg.get("model_dir") or None
        if model_dir:
            print(f"[T3 Train] Loading from local dir: {model_dir}")
            self.cb_model = ChatterboxTTS.from_local(model_dir, str(self.device))
        else:
            print("[T3 Train] Downloading ChatterboxTTS from HuggingFace...")
            self.cb_model = ChatterboxTTS.from_pretrained(str(self.device))

        self.t3 = self.cb_model.t3

        # ----- Apply LoRA -----
        self.t3 = apply_lora_to_t3(
            self.t3,
            lora_r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
        )
        self.t3.to(self.device)
        self.t3.train()

        # ----- Optimiser -----
        trainable_params = [p for p in self.t3.parameters() if p.requires_grad]
        self.optim = optim.AdamW(
            trainable_params,
            lr=cfg.get("t3_lr", 2e-4),
            weight_decay=0.01,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=cfg["num_epochs"] * len(self.loader),
        )

        # AMP
        self.use_amp = cfg.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Output
        self.output_dir = Path(cfg.get("t3_output_dir", cfg.get("checkpoint_dir", "./checkpoints/t3_lora")))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss weights
        self.speech_loss_weight = cfg.get("speech_loss_weight", 1.0)
        self.text_loss_weight = cfg.get("text_loss_weight", 0.1)

    def train(self):
        cfg = self.cfg
        best_loss = float("inf")

        print(f"\n[T3 Train] Starting LoRA fine-tuning for {cfg['num_epochs']} epochs")
        print(f"  Samples : {len(self.dataset)}")
        print(f"  Device  : {self.device}")
        print(f"  AMP     : {self.use_amp}")
        print(f"  Output  : {self.output_dir}\n")

        for epoch in range(1, cfg["num_epochs"] + 1):
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for batch in self.loader:
                try:
                    (t3_cond, txt_tok, txt_lens,
                     spch_tok, spch_lens) = collate_t3_batch(
                        batch, self.cb_model, self.device
                    )
                except Exception as e:
                    import traceback
                    print(f"  [WARN] Skipping batch: {e}")
                    traceback.print_exc()
                    continue

                self.optim.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    loss_text, loss_speech = self.t3.loss(
                        t3_cond=t3_cond,
                        text_tokens=txt_tok,
                        text_token_lens=txt_lens,
                        speech_tokens=spch_tok,
                        speech_token_lens=spch_lens,
                    )
                    loss = (self.text_loss_weight * loss_text +
                            self.speech_loss_weight * loss_speech)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.t3.parameters() if p.requires_grad], 1.0
                )
                self.scaler.step(self.optim)
                self.scaler.update()
                self.scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{cfg['num_epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_lora_model(self.t3, self.output_dir / "best", merge=False)

            save_every = cfg.get("save_every_n_epochs", 5)
            if epoch % save_every == 0:
                save_lora_model(self.t3, self.output_dir / f"epoch_{epoch:03d}", merge=False)

        # Final: merge + save full model
        print("\n[T3 Train] Merging LoRA weights for deployment...")
        save_lora_model(self.t3, self.output_dir / "final_merged", merge=True)
        print(f"\n[T3 Train] Done! Best loss: {best_loss:.4f}")


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune T3 with LoRA")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--model_dir", type=str, default=None, help="Local Chatterbox model dir")
    p.add_argument("--output_dir", type=str, default="./checkpoints/t3_lora")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--t3_batch_size", type=int, default=2)
    p.add_argument("--t3_lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_config(args) -> dict:
    cfg = {}
    if args.config:
        cfg = load_yaml(args.config)
    for key in ["data_dir", "model_dir", "output_dir", "num_epochs", "t3_batch_size",
                "t3_lr", "lora_r", "lora_alpha", "use_amp", "device"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg.setdefault(key, val)
    cfg.setdefault("t3_output_dir", cfg.pop("output_dir", "./checkpoints/t3_lora"))
    cfg.setdefault("checkpoint_dir", cfg.get("t3_output_dir"))
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)

    if not config.get("data_dir"):
        print("ERROR: --data_dir is required (or set data_dir in train_config.yaml)")
        sys.exit(1)

    trainer = T3LoRATrainer(config)
    trainer.train()
