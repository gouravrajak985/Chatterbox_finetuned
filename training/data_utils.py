"""
Data utilities for fine-tuning Chatterbox TTS models.

Folder structure expected:
    <data_dir>/
        speaker_A/
            audio1.wav
            audio2.mp3
            ...
        speaker_B/
            audio1.wav
            ...

If only ONE speaker folder exists (single-speaker mode), the dataset still works
for VoiceEncoder fine-tuning but GE2E contrast will be limited.
For T3 fine-tuning, audio files must be paired with transcripts (same basename .txt).
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
#  Constants                                                                    #
# --------------------------------------------------------------------------- #
VE_SR = 16_000      # VoiceEncoder sample rate (16 kHz)
S3_SR = 16_000      # S3Tokenizer sample rate  (same)
S3GEN_SR = 24_000   # S3Gen / vocoder sample rate
CLIP_DURATION = 3.0  # seconds per clip for VoiceEncoder training


# --------------------------------------------------------------------------- #
#  Resampling helper                                                            #
# --------------------------------------------------------------------------- #

def load_and_resample(fpath: str, target_sr: int) -> np.ndarray:
    """Load any audio file and resample to target_sr. Returns float32 mono."""
    wav, sr = librosa.load(fpath, sr=target_sr, mono=True)
    return wav.astype(np.float32)


# --------------------------------------------------------------------------- #
#  VoiceEncoder Dataset  (clips from same speaker are grouped)                 #
# --------------------------------------------------------------------------- #

class VoiceEncoderDataset(Dataset):
    """
    Produces fixed-length audio clips for GE2E / VoiceEncoder fine-tuning.

    Each item: (clip_tensor [samples], speaker_id [int])
    """

    def __init__(
        self,
        data_dir: str,
        clip_duration: float = CLIP_DURATION,
        sample_rate: int = VE_SR,
        min_clips_per_speaker: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.clip_len = int(clip_duration * sample_rate)

        # Discover speakers → files
        self.speaker_to_files: Dict[str, List[Path]] = {}
        for speaker_dir in sorted(self.data_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            files = [
                f for f in speaker_dir.rglob("*")
                if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            ]
            if len(files) >= min_clips_per_speaker:
                self.speaker_to_files[speaker_dir.name] = files

        if not self.speaker_to_files:
            raise ValueError(
                f"No speaker folders with audio found in {data_dir}. "
                "Make sure sub-folders contain .wav/.mp3 files."
            )

        self.speakers: List[str] = sorted(self.speaker_to_files.keys())
        self.speaker_to_id: Dict[str, int] = {s: i for i, s in enumerate(self.speakers)}

        # Flatten all clips into (fpath, speaker_id) pairs
        self.samples: List[Tuple[Path, int]] = []
        for spk, files in self.speaker_to_files.items():
            spk_id = self.speaker_to_id[spk]
            for f in files:
                self.samples.append((f, spk_id))

        print(
            f"[VoiceEncoderDataset] {len(self.speakers)} speakers, "
            f"{len(self.samples)} audio files in {data_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fpath, spk_id = self.samples[idx]
        wav = load_and_resample(str(fpath), self.sample_rate)

        # Take a random fixed-length crop
        if len(wav) >= self.clip_len:
            start = random.randint(0, len(wav) - self.clip_len)
            wav = wav[start: start + self.clip_len]
        else:
            # Pad with silence if clip is shorter than required
            pad = self.clip_len - len(wav)
            wav = np.pad(wav, (0, pad), mode="constant")

        return torch.from_numpy(wav), spk_id

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)


# --------------------------------------------------------------------------- #
#  T3 Dataset  (text + audio pairs)                                            #
# --------------------------------------------------------------------------- #

class T3FineTuneDataset(Dataset):
    """
    Dataset for T3 (text-to-speech-token) fine-tuning.

    Expects paired files:
        speaker_dir/utterance.wav   (audio)
        speaker_dir/utterance.txt   (transcript)

    Each item: (text_str, audio_path, speaker_name)
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples: List[Tuple[str, str, str]] = []  # (text, audio_path, speaker_name)

        for speaker_dir in sorted(self.data_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            audio_files = [
                f for f in speaker_dir.rglob("*")
                if f.suffix.lower() in {".wav", ".mp3", ".flac"}
            ]
            for af in audio_files:
                txt_path = af.with_suffix(".txt")
                if txt_path.exists():
                    text = txt_path.read_text(encoding="utf-8").strip()
                    if text:
                        self.samples.append((text, str(af), speaker_dir.name))

        if not self.samples:
            raise ValueError(
                f"No paired (audio + .txt) files found in {data_dir}.\n"
                "Each audio file must have a matching .txt file with the same name."
            )

        print(f"[T3FineTuneDataset] {len(self.samples)} paired samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        text, audio_path, speaker_name = self.samples[idx]
        return {
            "text": text,
            "audio_path": audio_path,
            "speaker_name": speaker_name,
        }


# --------------------------------------------------------------------------- #
#  DataLoader factories                                                         #
# --------------------------------------------------------------------------- #

def get_ve_dataloader(
    data_dir: str,
    batch_size: int = 64,
    clip_duration: float = CLIP_DURATION,
    num_workers: int = 4,
    shuffle: bool = True,
) -> Tuple[DataLoader, VoiceEncoderDataset]:
    """Returns (DataLoader, dataset) for VoiceEncoder fine-tuning."""
    dataset = VoiceEncoderDataset(data_dir, clip_duration=clip_duration)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset


def get_t3_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
) -> Tuple[DataLoader, T3FineTuneDataset]:
    """
    Returns (DataLoader, dataset) for T3 fine-tuning.
    Note: collation is handled by the training script (variable-length sequences).
    """
    dataset = T3FineTuneDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_t3_collate_fn,
    )
    return loader, dataset


def _t3_collate_fn(batch: List[Dict]) -> Dict:
    """Collate T3 samples into a batch dict."""
    return {
        "texts": [item["text"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
        "speaker_names": [item["speaker_name"] for item in batch],
    }


# --------------------------------------------------------------------------- #
#  Audio preprocessing utilities                                               #
# --------------------------------------------------------------------------- #

def preprocess_audio_folder(
    input_dir: str,
    output_dir: str,
    target_sr: int = VE_SR,
    min_duration_s: float = 1.0,
    max_duration_s: float = 30.0,
):
    """
    Preprocess all audio files in a directory tree:
    - Resample to target_sr
    - Skip files that are too short or too long
    - Save as .wav in mirrored output_dir structure

    Args:
        input_dir:      Root folder with speaker sub-folders.
        output_dir:     Where to write preprocessed files.
        target_sr:      Target sample rate.
        min_duration_s: Skip clips shorter than this.
        max_duration_s: Skip clips longer than this.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    total, kept, skipped = 0, 0, 0

    for audio_file in input_dir.rglob("*"):
        if audio_file.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            continue
        total += 1
        try:
            wav, sr = librosa.load(str(audio_file), sr=None, mono=True)
            duration = len(wav) / sr
            if duration < min_duration_s or duration > max_duration_s:
                skipped += 1
                continue
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            rel = audio_file.relative_to(input_dir)
            out_path = output_dir / rel.with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf
            sf.write(str(out_path), wav.astype(np.float32), target_sr)
            kept += 1
        except Exception as e:
            print(f"  [WARN] Skipping {audio_file}: {e}")
            skipped += 1

    print(
        f"[preprocess_audio_folder] Done. "
        f"Total: {total}, Kept: {kept}, Skipped: {skipped}"
    )
