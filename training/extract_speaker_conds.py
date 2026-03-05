"""
Extract and Save Speaker Conditionals
======================================
After fine-tuning the VoiceEncoder (and optionally T3), run this script to:
  1. Load the updated model weights
  2. Process a reference audio file for the new speaker
  3. Save the resulting Conditionals as a named speaker profile

Usage:
    python training/extract_speaker_conds.py \\
        --audio_path ./my_speaker_data/speaker_Alice/reference_clip.wav \\
        --speaker_name "Alice" \\
        --ve_ckpt ./checkpoints/ve_best.safetensors \\
        --t3_ckpt ./checkpoints/t3_lora/final_merged/t3_cfg.safetensors \\
        --speakers_dir ./speakers

After this, "Alice" will appear in the Gradio app's speaker dropdown.
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatterbox.tts import ChatterboxTTS
from chatterbox.speaker_manager import SpeakerManager


def extract_and_save_speaker(
    audio_path: str,
    speaker_name: str,
    ve_ckpt: str = None,
    t3_ckpt: str = None,
    model_dir: str = None,
    speakers_dir: str = "./speakers",
    exaggeration: float = 0.5,
    device: str = "cuda",
):
    """
    Load Chatterbox, optionally swap in fine-tuned weights, extract speaker
    Conditionals from a reference audio clip, and save to the speaker library.

    Args:
        audio_path:    Path to reference audio file (10–30s recommended).
        speaker_name:  Name to save this speaker as.
        ve_ckpt:       Optional path to fine-tuned ve.safetensors.
        t3_ckpt:       Optional path to fine-tuned t3_cfg.safetensors.
        model_dir:     Local Chatterbox model directory (skips HF download).
        speakers_dir:  Directory to save speaker profiles.
        exaggeration:  Emotion exaggeration value (0.5 = neutral).
        device:        Torch device string.
    """
    print(f"\n[ExtractConds] Speaker: '{speaker_name}'")
    print(f"[ExtractConds] Audio  : {audio_path}")

    # ----- Load base model -----
    if model_dir:
        print(f"[ExtractConds] Loading from local dir: {model_dir}")
        model = ChatterboxTTS.from_local(model_dir, device)
    else:
        print("[ExtractConds] Downloading from HuggingFace (or using cache)...")
        model = ChatterboxTTS.from_pretrained(device)

    # ----- Swap in fine-tuned VE -----
    if ve_ckpt:
        ve_path = Path(ve_ckpt)
        if ve_path.exists():
            print(f"[ExtractConds] Loading fine-tuned VE: {ve_path}")
            state = load_file(str(ve_path))
            model.ve.load_state_dict(state)
            model.ve.to(device).eval()
        else:
            print(f"[ExtractConds] WARNING: ve_ckpt not found ({ve_path}), using original VE.")

    # ----- Swap in fine-tuned T3 -----
    if t3_ckpt:
        t3_path = Path(t3_ckpt)
        if t3_path.exists():
            print(f"[ExtractConds] Loading fine-tuned T3: {t3_path}")
            state = load_file(str(t3_path))
            if "model" in state:
                state = state["model"][0]
            model.t3.load_state_dict(state, strict=False)
            model.t3.to(device).eval()
        else:
            print(f"[ExtractConds] WARNING: t3_ckpt not found ({t3_path}), using original T3.")

    # ----- Extract conditionals -----
    print("[ExtractConds] Extracting speaker conditionals from audio...")
    model.prepare_conditionals(audio_path, exaggeration=exaggeration)

    # ----- Save to speaker library -----
    sm = SpeakerManager(speakers_dir)
    saved_path = sm.save_speaker(speaker_name, model.conds)
    print(f"\n[ExtractConds] ✅ Speaker '{speaker_name}' saved to: {saved_path}")
    print(f"[ExtractConds] Available speakers: {sm.list_speakers()}")
    return saved_path


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Extract and save a speaker profile for Chatterbox TTS")
    p.add_argument("--audio_path", type=str, required=True,
                   help="Path to reference audio file (10-30s recommended)")
    p.add_argument("--speaker_name", type=str, required=True,
                   help="Name for this speaker in the library")
    p.add_argument("--ve_ckpt", type=str, default=None,
                   help="Path to fine-tuned ve.safetensors (from train_ve.py)")
    p.add_argument("--t3_ckpt", type=str, default=None,
                   help="Path to fine-tuned t3_cfg.safetensors (from train_t3.py)")
    p.add_argument("--model_dir", type=str, default=None,
                   help="Local Chatterbox model dir (optional, skips HF download)")
    p.add_argument("--speakers_dir", type=str, default="./speakers",
                   help="Directory to save speaker profiles")
    p.add_argument("--exaggeration", type=float, default=0.5,
                   help="Emotion exaggeration (0.5 = neutral)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_and_save_speaker(
        audio_path=args.audio_path,
        speaker_name=args.speaker_name,
        ve_ckpt=args.ve_ckpt,
        t3_ckpt=args.t3_ckpt,
        model_dir=args.model_dir,
        speakers_dir=args.speakers_dir,
        exaggeration=args.exaggeration,
        device=args.device,
    )
