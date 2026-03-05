"""
Speaker Manager for Chatterbox TTS
Manages named speaker voice profiles (pre-computed Conditionals).
Each profile is saved as a .pt file in the speakers directory.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List

# Conditionals will be imported lazily to avoid circular imports
# It is already defined in chatterbox.tts


class SpeakerManager:
    """
    Manages a library of named speaker voice profiles.

    Usage:
        sm = SpeakerManager("./speakers")
        # Save a speaker (from ChatterboxTTS.conds after prepare_conditionals)
        sm.save_speaker("alice", model.conds)
        # Load a speaker
        conds = sm.load_speaker("alice")
        # List all speakers
        names = sm.list_speakers()
        # Delete a speaker
        sm.delete_speaker("alice")
    """

    EXTENSION = ".pt"

    def __init__(self, speaker_dir: str = "./speakers"):
        self.speaker_dir = Path(speaker_dir)
        self.speaker_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  CRUD Operations                                                      #
    # ------------------------------------------------------------------ #

    def save_speaker(self, name: str, conds) -> Path:
        """
        Save a speaker's Conditionals object to disk.

        Args:
            name:  Human-readable speaker name (used as filename).
            conds: A `Conditionals` instance from chatterbox.tts.

        Returns:
            Path to the saved file.
        """
        name = self._sanitize_name(name)
        fpath = self._fpath(name)
        conds.save(fpath)
        print(f"[SpeakerManager] Saved speaker '{name}' → {fpath}")
        return fpath

    def load_speaker(self, name: str, map_location: str = "cpu"):
        """
        Load a speaker's Conditionals from disk.

        Args:
            name:         Speaker name.
            map_location: Torch device string, default 'cpu'.

        Returns:
            Conditionals object.

        Raises:
            FileNotFoundError: if the speaker profile doesn't exist.
        """
        from chatterbox.tts import Conditionals  # lazy import

        name = self._sanitize_name(name)
        fpath = self._fpath(name)
        if not fpath.exists():
            raise FileNotFoundError(
                f"Speaker '{name}' not found. Available: {self.list_speakers()}"
            )
        conds = Conditionals.load(fpath, map_location=map_location)
        print(f"[SpeakerManager] Loaded speaker '{name}' from {fpath}")
        return conds

    def list_speakers(self) -> List[str]:
        """Return sorted list of saved speaker names."""
        return sorted(
            p.stem for p in self.speaker_dir.glob(f"*{self.EXTENSION}")
        )

    def delete_speaker(self, name: str) -> bool:
        """
        Delete a speaker profile.

        Returns:
            True if deleted, False if not found.
        """
        name = self._sanitize_name(name)
        fpath = self._fpath(name)
        if fpath.exists():
            fpath.unlink()
            print(f"[SpeakerManager] Deleted speaker '{name}'")
            return True
        print(f"[SpeakerManager] Speaker '{name}' not found, nothing deleted.")
        return False

    def speaker_exists(self, name: str) -> bool:
        """Check if a speaker profile exists."""
        return self._fpath(self._sanitize_name(name)).exists()

    def rename_speaker(self, old_name: str, new_name: str) -> Path:
        """Rename a speaker profile."""
        old_name = self._sanitize_name(old_name)
        new_name = self._sanitize_name(new_name)
        old_path = self._fpath(old_name)
        new_path = self._fpath(new_name)
        if not old_path.exists():
            raise FileNotFoundError(f"Speaker '{old_name}' not found.")
        if new_path.exists():
            raise FileExistsError(f"Speaker '{new_name}' already exists.")
        shutil.move(str(old_path), str(new_path))
        print(f"[SpeakerManager] Renamed '{old_name}' → '{new_name}'")
        return new_path

    def get_speaker_info(self, name: str) -> dict:
        """Return metadata about a speaker profile."""
        name = self._sanitize_name(name)
        fpath = self._fpath(name)
        if not fpath.exists():
            raise FileNotFoundError(f"Speaker '{name}' not found.")
        stat = fpath.stat()
        return {
            "name": name,
            "path": str(fpath),
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _fpath(self, name: str) -> Path:
        return self.speaker_dir / f"{name}{self.EXTENSION}"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Replace spaces / unsafe chars with underscores."""
        import re
        name = name.strip()
        name = re.sub(r"[^\w\-]", "_", name)
        if not name:
            raise ValueError("Speaker name cannot be empty.")
        return name
