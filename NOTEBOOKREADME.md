# Running & Training Chatterbox in a Modal Notebook

> **Platform:** [modal.com](https://modal.com) — Jupyter notebook with A100 GPU  
> **Cost:** ~$1.10/hr (billed per second)

---

## 1. Open a Notebook

1. Sign in at **modal.com**
2. Sidebar → **Notebooks** → **New Notebook**
3. Select **A100** GPU

---

## 2. Set Up the Environment

```python
# Cell 1 — Clone repo
!git clone https://github.com/YOUR_USERNAME/Chatterbox.git
%cd Chatterbox
!pip install -e . -q
!pip install peft omegaconf -q
```

```python
# Cell 2 — Set up Python paths (run this before ALL imports)
import sys, os
os.chdir("/root/Chatterbox")          # ← adjust if your folder name differs
repo_root = os.getcwd()
sys.path.insert(0, repo_root)          # for training.*
sys.path.insert(0, repo_root + "/src") # for chatterbox.*
print("Working dir:", repo_root)
```

---

## 3. Upload Your Speaker Data

Your data must follow this folder structure **after extraction**:
```
my_speaker_data/
    speaker_name/
        clip001.wav
        clip002.wav
        ...
```

**If your data is a zip file:**
```python
# Cell 3 — Unzip data
import zipfile, os

zip_path = "training/my_speaker_data/your_data.zip"  # ← update filename
extract_to = "training/my_speaker_data"

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_to)

# Verify
for root, dirs, files in os.walk(extract_to):
    if files:
        print(root, "→", len(files), "files")
        break
```

**If uploading from Google Drive:**
```python
!pip install gdown
!gdown --folder YOUR_GDRIVE_FOLDER_ID -O training/my_speaker_data
```

---

## 4A. Train VoiceEncoder (recommended first)

Fine-tunes the speaker identity encoder.

```python
import sys, os
os.chdir("/root/Chatterbox")           # skip if already run in Cell 2
repo_root = os.getcwd()
sys.path.insert(0, repo_root)
sys.path.insert(0, repo_root + "/src")

from training.train_ve import VoiceEncoderTrainer

config = {
    "data_dir":            "training/my_speaker_data",
    "checkpoint_dir":      "checkpoints/ve",
    "num_epochs":          30,
    "batch_size":          64,
    "lr":                  1e-4,
    "clips_per_speaker":   10,
    "device":              "cuda",
    "use_amp":             True,
    "num_workers":         4,
    "save_every_n_epochs": 5,
}

trainer = VoiceEncoderTrainer(config)
trainer.train()
```

**Output:** `checkpoints/ve/ve_best.safetensors`

---

## 4B. Train T3 with LoRA

Fine-tunes the T3 language model for your speaker.  
**Requires paired `.wav` + `.txt` files** (transcript for each audio clip).

```python
import sys, os
os.chdir("/root/Chatterbox")           # skip if already run in Cell 2
repo_root = os.getcwd()
sys.path.insert(0, repo_root)
sys.path.insert(0, repo_root + "/src")

from training.train_t3 import T3LoRATrainer

config = {
    "data_dir":            "training/my_speaker_data",
    "checkpoint_dir":      "checkpoints/t3_lora",
    "t3_output_dir":       "checkpoints/t3_lora",
    "num_epochs":          10,
    "t3_batch_size":       2,
    "t3_lr":               2e-4,
    "lora_r":              16,
    "lora_alpha":          32,
    "device":              "cuda",
    "use_amp":             True,
    "num_workers":         0,
    "save_every_n_epochs": 2,
}

trainer = T3LoRATrainer(config)
trainer.train()
```

**Output:** `checkpoints/t3_lora/best/` (LoRA adapter)

---

## 5. Merge LoRA into Base Model

Run after T3 training to produce a deployable `t3_cfg.safetensors`:

```python
!python training/merge_lora.py \
    --lora_dir checkpoints/t3_lora/best \
    --output_dir checkpoints/t3_lora/final_merged
```

---

## 6. Download Checkpoints

```python
import shutil

# Zip all checkpoints
shutil.make_archive("chatterbox_checkpoints", "zip", "checkpoints")
print("Ready to download: chatterbox_checkpoints.zip")
# Download from the notebook file browser (left sidebar → right-click → Download)
```

---

## Folder Structure After Training

```
checkpoints/
    ve/
        ve_best.safetensors       ← best VoiceEncoder
        ve_final.safetensors      ← final VoiceEncoder
    t3_lora/
        best/
            lora_adapter/         ← LoRA weights only
            t3_extra_params.pt    ← embeddings & heads
        final_merged/
            t3_cfg.safetensors    ← merged, ready to deploy ✅
```

---

## Tips

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: training` | Make sure `sys.path.insert(0, ".")` is run |
| Zip nested folder | Update `data_dir` to the correct subfolder path |
| Session timeout | Use `modal run` CLI for long runs (no browser required) |
| Out of memory | Reduce `batch_size` or `t3_batch_size` |
