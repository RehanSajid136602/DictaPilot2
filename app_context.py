"""
DictaPilot app context & per-app profiles.

Original by: Rohan Sharvesh
Fork maintained by: Rehan

MIT License
Copyright (c) 2026 Rohan Sharvesh
Copyright (c) 2026 Rehan

Profiles file format (JSON):
{
  "default": {"tone": "polite", "language": "english"},
  "apps": {
    "Slack": {"tone": "casual"},
    "Gmail": {"tone": "formal", "language": "english"}
  }
}
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DictationContext:
    app_id: Optional[str]
    tone: str
    language: str


DEFAULT_TONE = (os.getenv("DEFAULT_TONE") or "polite").strip().lower()
DEFAULT_LANGUAGE = (os.getenv("DEFAULT_LANGUAGE") or "english").strip().lower()


def _profiles_path() -> Path:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("APPDATA") or ""
        return Path(base) / "DictaPilot" / "profiles.json"
    return Path.home() / ".config" / "dictapilot" / "profiles.json"


_PROFILE_CACHE = {"path": None, "mtime": None, "data": {}}


def _load_profiles() -> dict:
    path = _profiles_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _PROFILE_CACHE.update({"path": path_str, "mtime": None, "data": {}})
        return {}

    if _PROFILE_CACHE.get("path") == path_str and _PROFILE_CACHE.get("mtime") == mtime:
        return _PROFILE_CACHE.get("data", {})

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    _PROFILE_CACHE.update({"path": path_str, "mtime": mtime, "data": data})
    return data


def _save_profiles(data: dict) -> None:
    path = _profiles_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _PROFILE_CACHE.update({"path": str(path), "mtime": path.stat().st_mtime, "data": data})


def update_profile(app_id: Optional[str], tone: Optional[str] = None, language: Optional[str] = None) -> None:
    data = _load_profiles()
    if app_id:
        apps = data.setdefault("apps", {})
        profile = apps.setdefault(app_id, {})
    else:
        profile = data.setdefault("default", {})
    if tone:
        profile["tone"] = tone
    if language:
        profile["language"] = language
    _save_profiles(data)


def get_context() -> DictationContext:
    app_id = _active_app_id()
    data = _load_profiles()
    default = data.get("default") or {}
    tone = (default.get("tone") or DEFAULT_TONE).strip().lower()
    language = (default.get("language") or DEFAULT_LANGUAGE).strip().lower()

    if app_id:
        app_cfg = _match_profile(app_id, data.get("apps") or {})
        if app_cfg:
            tone = (app_cfg.get("tone") or tone).strip().lower()
            language = (app_cfg.get("language") or language).strip().lower()

    return DictationContext(app_id=app_id, tone=tone, language=language)


def _match_profile(app_id: str, apps: dict) -> Optional[dict]:
    if not app_id:
        return None
    lower = app_id.lower()
    for key, profile in apps.items():
        if key and key.lower() in lower:
            return profile
    return None


def _active_app_id() -> Optional[str]:
    override = os.getenv("ACTIVE_APP")
    if override:
        return override.strip()

    system = platform.system()
    if system == "Windows":
        return _active_app_windows()
    if system == "Darwin":
        return _active_app_macos()
    if system == "Linux":
        return _active_app_linux()
    return None


def _active_app_macos() -> Optional[str]:
    script = 'tell application "System Events" to get name of first application process whose frontmost is true'
    try:
        result = subprocess.check_output(["osascript", "-e", script], timeout=0.5)
        return result.decode("utf-8", "ignore").strip() or None
    except Exception:
        return None


def _active_app_linux() -> Optional[str]:
    if shutil.which("xdotool"):
        try:
            result = subprocess.check_output(
                ["xdotool", "getactivewindow", "getwindowname"],
                timeout=0.5,
            )
            return result.decode("utf-8", "ignore").strip() or None
        except Exception:
            return None
    return None


def _active_app_windows() -> Optional[str]:
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process = kernel32.OpenProcess(0x0410, False, pid.value)
        if not process:
            return None

        buf = (wintypes.WCHAR * 260)()
        psapi.GetModuleBaseNameW(process, None, buf, 260)
        kernel32.CloseHandle(process)
        name = buf.value.strip()
        return name or None
    except Exception:
        return None
