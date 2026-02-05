"""
DictaPilot configuration module
Handles config.json persistence and environment variable overrides

Original by: Rohan Sharvesh
Fork maintained by: Rehan

MIT License
Copyright (c) 2026 Rohan Sharvesh
Copyright (c) 2026 Rehan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import json
import platform
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict


def get_config_path() -> Path:
    """Get platform-specific config file path"""
    system = platform.system()
    
    if system == "Windows":
        config_dir = Path(os.environ.get("APPDATA", ""))
        return config_dir / "DictaPilot" / "config.json"
    else:
        config_dir = Path(os.path.expanduser("~/.config"))
        return config_dir / "dictapilot" / "config.json"


def get_config_dir() -> Path:
    """Create and return config directory"""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    return config_path.parent


DEFAULT_CONFIG = {
    "hotkey": "f9",
    "model": "whisper-large-v3-turbo",
    "smart_mode": "llm",
    "smart_edit": True,
    "llm_always_clean": True,
    "paste_mode": "delta",
    "paste_backend": "auto",
    "hotkey_backend": "auto",
    "reset_transcript_each_recording": True,
}


@dataclass
class DictaPilotConfig:
    hotkey: str = "f9"
    model: str = "whisper-large-v3-turbo"
    smart_mode: str = "llm"
    smart_edit: bool = True
    llm_always_clean: bool = True
    paste_mode: str = "delta"
    paste_backend: str = "auto"
    hotkey_backend: str = "auto"
    reset_transcript_each_recording: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DictaPilotConfig":
        if data is None:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save config to JSON file"""
        path = path or get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DictaPilotConfig":
        """Load config from JSON file, return defaults if not exists"""
        path = path or get_config_path()
        if not path.exists():
            return cls()
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return cls()


def load_config() -> DictaPilotConfig:
    """
    Load configuration with environment variable overrides
    Priority: CLI args > env vars > config file > defaults
    """
    config = DictaPilotConfig.load()
    
    env_overrides = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "HOTKEY": os.getenv("HOTKEY"),
        "GROQ_WHISPER_MODEL": os.getenv("GROQ_WHISPER_MODEL"),
        "GROQ_CHAT_MODEL": os.getenv("GROQ_CHAT_MODEL"),
        "SMART_EDIT": os.getenv("SMART_EDIT"),
        "SMART_MODE": os.getenv("SMART_MODE"),
        "LLM_ALWAYS_CLEAN": os.getenv("LLM_ALWAYS_CLEAN"),
        "PASTE_MODE": os.getenv("PASTE_MODE"),
        "PASTE_BACKEND": os.getenv("PASTE_BACKEND"),
        "HOTKEY_BACKEND": os.getenv("HOTKEY_BACKEND"),
        "RESET_TRANSCRIPT_EACH_RECORDING": os.getenv("RESET_TRANSCRIPT_EACH_RECORDING"),
    }
    
    for key, value in env_overrides.items():
        if value is not None:
            if key == "HOTKEY":
                config.hotkey = value
            elif key == "GROQ_WHISPER_MODEL":
                config.model = value
            elif key == "GROQ_CHAT_MODEL":
                pass
            elif key == "SMART_EDIT":
                config.smart_edit = value.lower() in ("1", "true", "yes")
            elif key == "SMART_MODE":
                config.smart_mode = value.lower()
            elif key == "LLM_ALWAYS_CLEAN":
                config.llm_always_clean = value.lower() in ("1", "true", "yes")
            elif key == "PASTE_MODE":
                config.paste_mode = value.lower()
            elif key == "PASTE_BACKEND":
                config.paste_backend = value.lower()
            elif key == "HOTKEY_BACKEND":
                config.hotkey_backend = value.lower()
            elif key == "RESET_TRANSCRIPT_EACH_RECORDING":
                config.reset_transcript_each_recording = value.lower() in ("1", "true", "yes")
    
    return config


def apply_config_to_env(config: DictaPilotConfig) -> None:
    """Apply config settings to environment variables"""
    os.environ["HOTKEY"] = config.hotkey
    os.environ["GROQ_WHISPER_MODEL"] = config.model
    os.environ["SMART_MODE"] = config.smart_mode
    os.environ["SMART_EDIT"] = "1" if config.smart_edit else "0"
    os.environ["LLM_ALWAYS_CLEAN"] = "1" if config.llm_always_clean else "0"
    os.environ["PASTE_MODE"] = config.paste_mode
    os.environ["PASTE_BACKEND"] = config.paste_backend
    os.environ["HOTKEY_BACKEND"] = config.hotkey_backend
    os.environ["RESET_TRANSCRIPT_EACH_RECORDING"] = "1" if config.reset_transcript_each_recording else "0"
