# Training Pipeline for Chatterbox TTS

This folder contains scripts to fine-tune Chatterbox TTS on your own speaker data (10+ hours) and add new speakers to the speaker library.

---

## Overview

```
training/
├── data_utils.py            ← Audio dataset & preprocessing
├── train_ve.py              ← Step 1: Fine-tune VoiceEncoder (GE2E loss)
├── train_t3.py              ← Step 2: Fine-tune T3 LLM with LoRA
├── extract_speaker_conds.py ← Step 3: Export speaker profile
└── train_config.yaml        ← All hyperparameters (edit this!)
```

---

## Prerequisites

```bash
pip install peft pyyaml soundfile
# For training: CUDA GPU strongly recommended
```

---

## Step 0 — Prepare Your Data

### For VoiceEncoder training (any audio, no transcripts needed):
```
my_speaker_data/
    speaker_Alice/
        clip001.wav
        clip002.wav
        ...
    speaker_Bob/          ← (optional) more speakers = better discrimination
        clip001.wav
        ...
```

### For T3 fine-tuning (paired audio + transcripts required):
```
my_speaker_data/
    speaker_Alice/
        utterance001.wav
        utterance001.txt   ← exact transcription of the wav file
        utterance002.wav
        utterance002.txt
        ...
```

> **Tip**: Tools like [Whisper](https://github.com/openai/whisper) can auto-generate transcripts:
> ```bash
> pip install openai-whisper
> whisper my_audio.wav --model medium --output_dir ./speaker_Alice/
> # Rename the .txt output to match the wav filename
> ```

---

## Step 1 — Configure Training

Edit `training/train_config.yaml`:

```yaml
data_dir: "./my_speaker_data"
device: "cuda"
use_amp: true
num_epochs: 30          # VE training epochs
t3_num_epochs: 10       # T3 LoRA epochs
```

---

## Step 2 — Fine-tune VoiceEncoder

VoiceEncoder learns speaker identity from audio clips. No transcripts needed.

```bash
cd path/to/Chatterbox

python training/train_ve.py --config training/train_config.yaml
```

Checkpoints are saved to `./checkpoints/`:
- `ve_best.safetensors` — best checkpoint
- `ve_final.safetensors` — last epoch

**Training time**: ~1–3 hours on a single GPU with 10+ hours of audio.

---

## Step 3 — Fine-tune T3 with LoRA (optional but recommended)

T3 is the 520M LLM that converts text → speech tokens. Fine-tuning with LoRA adapts it to your speaker's style with minimal VRAM.

```bash
python training/train_t3.py --config training/train_config.yaml
```

The merged model is saved to `./checkpoints/t3_lora/final_merged/t3_cfg.safetensors`.

> **VRAM requirements**: ~10 GB at batch_size=2 with AMP. Reduce `t3_batch_size` to 1 if needed.

---

## Step 4 — Export Speaker Profile

After training, extract and save the speaker's voice profile:

```bash
python training/extract_speaker_conds.py \
    --audio_path "./my_speaker_data/speaker_Alice/reference_clip.wav" \
    --speaker_name "Alice" \
    --ve_ckpt "./checkpoints/ve_best.safetensors" \
    --t3_ckpt "./checkpoints/t3_lora/final_merged/t3_cfg.safetensors" \
    --speakers_dir "./speakers"
```

This saves `./speakers/Alice.pt` — a pre-computed voice profile.

---

## Step 5 — Use in the Multi-Speaker Gradio App

```bash
python multi_speaker_gradio_app.py
```

Your speaker "Alice" will now appear in the **Generate** tab's speaker dropdown.

---

## Quick Flow (no training — just zero-shot cloning)

If you just want to save a speaker profile WITHOUT fine-tuning:

```bash
python training/extract_speaker_conds.py \
    --audio_path "./any_reference_audio.wav" \
    --speaker_name "MySpeaker" \
    --speakers_dir "./speakers"
```

---

## Tuning Tips

| Goal | Suggestion |
|------|-----------|
| Better speaker identity | More epochs for VE (`num_epochs: 50`), more data |
| More natural speech | More T3 epochs, lower `t3_lr` (1e-4) |
| Faster training | `use_amp: true`, larger `batch_size` |
| Low VRAM | Reduce `t3_batch_size: 1`, disable AMP |
| Better LoRA quality | Increase `lora_r: 32` |

---

## File Descriptions

| File | Purpose |
|------|---------|
| `data_utils.py` | Loads & preprocesses audio, creates PyTorch Datasets |
| `train_ve.py` | GE2E loss training for VoiceEncoder speaker embeddings |
| `train_t3.py` | LoRA fine-tuning for T3 (text-to-speech-token LLM) |
| `extract_speaker_conds.py` | Exports a `.pt` speaker profile to the speaker library |
| `train_config.yaml` | All training hyperparameters |
