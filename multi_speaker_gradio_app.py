"""
Multi-Speaker Gradio App for Chatterbox TTS
============================================
Features:
  - Generate Tab: pick a saved speaker OR upload new audio → generate TTS
  - Speaker Library Tab: upload audio → name it → save as speaker profile
                         list/delete saved speakers
  - All speaker profiles are stored in the ./speakers/ directory as .pt files

Run:
    python multi_speaker_gradio_app.py
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import gradio as gr

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chatterbox.tts import ChatterboxTTS
from chatterbox.speaker_manager import SpeakerManager


# --------------------------------------------------------------------------- #
#  Config                                                                       #
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPEAKERS_DIR = "./speakers"

# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
#  Model + Speaker Manager loader (called once at startup)                     #
# --------------------------------------------------------------------------- #

def load_model_and_sm():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    sm = SpeakerManager(SPEAKERS_DIR)
    return model, sm


# --------------------------------------------------------------------------- #
#  Generate Tab logic                                                            #
# --------------------------------------------------------------------------- #

def get_speaker_list(sm):
    """Return dropdown choices: saved speakers + '[ Upload New Audio ]' option."""
    saved = sm.list_speakers()
    choices = ["[ Upload New Audio ]"] + saved
    return choices


def generate_speech(
    model_state, sm_state,
    text,
    speaker_choice,
    upload_audio,
    exaggeration,
    cfg_weight,
    temperature,
    seed_num,
    min_p,
    top_p,
    repetition_penalty,
):
    model: ChatterboxTTS = model_state
    sm: SpeakerManager = sm_state

    if model is None:
        return None, "❌ Model not loaded yet, please wait."

    if not text or not text.strip():
        return None, "⚠️ Please enter some text."

    if seed_num != 0:
        set_seed(int(seed_num))

    status = ""

    # ----- Resolve speaker -----
    if speaker_choice and speaker_choice != "[ Upload New Audio ]":
        # Load saved speaker profile
        try:
            conds = sm.load_speaker(speaker_choice, map_location=DEVICE)
            model.conds = conds
            # Update exaggeration on the loaded conds
            from chatterbox.models.t3.modules.cond_enc import T3Cond
            _c = model.conds.t3
            model.conds.t3 = T3Cond(
                speaker_emb=_c.speaker_emb,
                cond_prompt_speech_tokens=_c.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(DEVICE)
            audio_prompt = None
            status = f"🎤 Using saved speaker: **{speaker_choice}**"
        except Exception as e:
            return None, f"❌ Could not load speaker '{speaker_choice}': {e}"
    elif upload_audio:
        audio_prompt = upload_audio
        status = "🎙️ Using uploaded audio as voice reference"
    else:
        return None, "⚠️ Please select a saved speaker OR upload a reference audio file."

    # ----- Generate -----
    try:
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return (model.sr, wav.squeeze(0).numpy()), status
    except Exception as e:
        return None, f"❌ Generation failed: {e}"


# --------------------------------------------------------------------------- #
#  Speaker Library Tab logic                                                    #
# --------------------------------------------------------------------------- #

def save_new_speaker(model_state, sm_state, audio_path, speaker_name):
    model: ChatterboxTTS = model_state
    sm: SpeakerManager = sm_state

    if not speaker_name or not speaker_name.strip():
        return "⚠️ Please enter a speaker name.", gr.update()
    if not audio_path:
        return "⚠️ Please upload an audio file.", gr.update()

    try:
        model.prepare_conditionals(audio_path, exaggeration=0.5)
        sm.save_speaker(speaker_name.strip(), model.conds)
        choices = get_speaker_list(sm)
        return (
            f"✅ Speaker **'{speaker_name.strip()}'** saved successfully!",
            gr.update(choices=choices, value=speaker_name.strip())
        )
    except Exception as e:
        return f"❌ Error: {e}", gr.update()


def delete_speaker(sm_state, speaker_name):
    sm: SpeakerManager = sm_state
    if not speaker_name or speaker_name == "[ Upload New Audio ]":
        return "⚠️ Select a speaker to delete.", gr.update(), gr.update()
    try:
        deleted = sm.delete_speaker(speaker_name)
        if deleted:
            choices = get_speaker_list(sm)
            return (
                f"🗑️ Speaker **'{speaker_name}'** deleted.",
                gr.update(choices=choices, value=choices[0] if choices else None),
                gr.update(choices=choices, value=choices[0] if choices else None),
            )
        return f"⚠️ Speaker '{speaker_name}' not found.", gr.update(), gr.update()
    except Exception as e:
        return f"❌ Error: {e}", gr.update(), gr.update()


def refresh_speaker_lists(sm_state):
    sm: SpeakerManager = sm_state
    choices = get_speaker_list(sm)
    lib_choices = sm.list_speakers()
    return (
        gr.update(choices=choices, value=choices[0] if choices else None),
        gr.update(choices=lib_choices, value=lib_choices[0] if lib_choices else None),
    )


def list_speakers_table(sm_state):
    sm: SpeakerManager = sm_state
    speakers = sm.list_speakers()
    if not speakers:
        return "No speakers saved yet."
    rows = []
    for name in speakers:
        info = sm.get_speaker_info(name)
        kb = info["size_bytes"] / 1024
        rows.append(f"| **{name}** | {kb:.1f} KB |")
    header = "| Speaker | Size |\n|---------|------|\n"
    return header + "\n".join(rows)


# --------------------------------------------------------------------------- #
#  Build Gradio UI                                                              #
# --------------------------------------------------------------------------- #

def build_ui():
    with gr.Blocks(
        title="Chatterbox TTS — Multi-Speaker",
        theme=gr.themes.Soft(primary_hue="violet"),
    ) as demo:
        model_state = gr.State(None)
        sm_state = gr.State(None)

        gr.Markdown(
            "# 🗣️ Chatterbox TTS — Multi-Speaker\n"
            "_Save speaker voice profiles and generate speech instantly._"
        )

        # ------------------------------------------------------------------ #
        #  Tab 1: Generate                                                      #
        # ------------------------------------------------------------------ #
        with gr.Tab("🎙️ Generate"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Text to synthesize",
                        placeholder="Type something here...",
                        lines=4,
                        max_lines=8,
                    )

                    with gr.Row():
                        speaker_dropdown = gr.Dropdown(
                            label="Saved Speaker",
                            choices=["[ Upload New Audio ]"],
                            value="[ Upload New Audio ]",
                            interactive=True,
                            scale=2,
                        )
                        refresh_btn = gr.Button("🔄 Refresh", scale=1, size="sm")

                    upload_audio = gr.Audio(
                        label="Reference Audio (used when '[ Upload New Audio ]' is selected)",
                        sources=["upload", "microphone"],
                        type="filepath",
                    )

                    with gr.Row():
                        exaggeration = gr.Slider(0.25, 2.0, step=0.05, value=0.5,
                                                  label="Exaggeration (Neutral = 0.5)")
                        cfg_weight = gr.Slider(0.0, 1.0, step=0.05, value=0.5,
                                               label="CFG / Pace")

                    with gr.Accordion("Advanced Options", open=False):
                        temperature = gr.Slider(0.05, 5.0, step=0.05, value=0.8, label="Temperature")
                        seed_num = gr.Number(value=0, label="Seed (0 = random)")
                        min_p = gr.Slider(0.0, 1.0, step=0.01, value=0.05, label="min_p")
                        top_p = gr.Slider(0.0, 1.0, step=0.01, value=1.0, label="top_p")
                        rep_pen = gr.Slider(1.0, 2.0, step=0.1, value=1.2, label="Repetition Penalty")

                    generate_btn = gr.Button("🔊 Generate", variant="primary", size="lg")

                with gr.Column(scale=2):
                    audio_output = gr.Audio(label="Output Audio", interactive=False)
                    status_text = gr.Markdown("_Ready_")

            generate_btn.click(
                fn=generate_speech,
                inputs=[
                    model_state, sm_state,
                    text_input, speaker_dropdown, upload_audio,
                    exaggeration, cfg_weight, temperature,
                    seed_num, min_p, top_p, rep_pen,
                ],
                outputs=[audio_output, status_text],
            )

        # ------------------------------------------------------------------ #
        #  Tab 2: Speaker Library                                               #
        # ------------------------------------------------------------------ #
        with gr.Tab("📚 Speaker Library"):
            gr.Markdown("### Add a New Speaker")
            with gr.Row():
                with gr.Column():
                    lib_upload = gr.Audio(
                        label="Speaker Reference Audio (10–60s recommended)",
                        sources=["upload", "microphone"],
                        type="filepath",
                    )
                    lib_name = gr.Textbox(label="Speaker Name", placeholder="e.g. Alice")
                    save_btn = gr.Button("💾 Save Speaker", variant="primary")
                    save_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Manage Speakers")
            with gr.Row():
                with gr.Column():
                    speaker_table = gr.Markdown("_Loading..._")
                    refresh_lib_btn = gr.Button("🔄 Refresh List")

                with gr.Column():
                    delete_dropdown = gr.Dropdown(
                        label="Select Speaker to Delete",
                        choices=[],
                        interactive=True,
                    )
                    delete_btn = gr.Button("🗑️ Delete Speaker", variant="stop")
                    delete_status = gr.Markdown("")

            save_btn.click(
                fn=save_new_speaker,
                inputs=[model_state, sm_state, lib_upload, lib_name],
                outputs=[save_status, speaker_dropdown],
            ).then(
                fn=lambda sm: (
                    list_speakers_table(sm),
                    gr.update(choices=sm.list_speakers()),
                ),
                inputs=[sm_state],
                outputs=[speaker_table, delete_dropdown],
            )

            refresh_lib_btn.click(
                fn=lambda sm: (list_speakers_table(sm), gr.update(choices=sm.list_speakers())),
                inputs=[sm_state],
                outputs=[speaker_table, delete_dropdown],
            )

            delete_btn.click(
                fn=delete_speaker,
                inputs=[sm_state, delete_dropdown],
                outputs=[delete_status, speaker_dropdown, delete_dropdown],
            ).then(
                fn=list_speakers_table,
                inputs=[sm_state],
                outputs=[speaker_table],
            )

            refresh_btn.click(
                fn=refresh_speaker_lists,
                inputs=[sm_state],
                outputs=[speaker_dropdown, delete_dropdown],
            )

        # ------------------------------------------------------------------ #
        #  Load model on startup                                                #
        # ------------------------------------------------------------------ #
        def startup():
            model, sm = load_model_and_sm()
            choices = get_speaker_list(sm)
            lib_choices = sm.list_speakers()
            return (
                model, sm,
                gr.update(choices=choices, value=choices[0] if choices else "[ Upload New Audio ]"),
                gr.update(choices=lib_choices),
                list_speakers_table(sm),
            )

        demo.load(
            fn=startup,
            inputs=[],
            outputs=[model_state, sm_state, speaker_dropdown, delete_dropdown, speaker_table],
        )

    return demo


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--speakers_dir", type=str, default="./speakers")
    parser.add_argument("--no-launch", action="store_true", help="Load model and exit (for testing)")
    args = parser.parse_args()

    SPEAKERS_DIR = args.speakers_dir

    demo = build_ui()

    if args.no_launch:
        # Dry-run: just check model loads without launching browser
        print("Dry-run: loading model...")
        model, sm = load_model_and_sm()
        print(f"Model OK. Speakers: {sm.list_speakers()}")
    else:
        demo.queue(max_size=50, default_concurrency_limit=1).launch(
            share=args.share,
            server_port=args.port,
        )
