"""
DictaPilot Paste Utilities
Cross-platform text injection for pasting transcribed text.

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

import shutil
import subprocess
import sys
from typing import List, Tuple


def longest_common_prefix(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def compute_delta(prev_text: str, new_text: str) -> Tuple[int, str]:
    lcp = longest_common_prefix(prev_text or "", new_text or "")
    to_delete = len(prev_text or "") - lcp
    to_insert = (new_text or "")[lcp:]
    return to_delete, to_insert


def _keyboard_module():
    import keyboard

    return keyboard


def _has_xdotool() -> bool:
    return shutil.which("xdotool") is not None


def _pynput_controller():
    from pynput.keyboard import Controller, Key, KeyCode

    return Controller(), Key, KeyCode


def _pynput_key_from_name(name: str):
    key_name = (name or "").strip().lower()
    _, Key, KeyCode = _pynput_controller()
    mapping = {
        "ctrl": Key.ctrl,
        "control": Key.ctrl,
        "cmd": Key.cmd,
        "command": Key.cmd,
        "alt": Key.alt,
        "shift": Key.shift,
        "tab": Key.tab,
        "enter": Key.enter,
        "return": Key.enter,
        "esc": Key.esc,
        "escape": Key.esc,
        "space": Key.space,
        "backspace": Key.backspace,
        "delete": Key.delete,
    }
    if key_name in mapping:
        return mapping[key_name]
    if key_name.startswith("f") and key_name[1:].isdigit() and hasattr(Key, key_name):
        return getattr(Key, key_name)
    if len(key_name) == 1:
        return KeyCode.from_char(key_name)
    return None


def _pynput_key(combo: str) -> bool:
    try:
        controller, _, _ = _pynput_controller()
        keys = [_pynput_key_from_name(part) for part in combo.split("+")]
        if any(key is None for key in keys):
            return False
        for key in keys:
            controller.press(key)
        for key in reversed(keys):
            controller.release(key)
        return True
    except Exception:
        return False


def _pynput_backspace(count: int) -> bool:
    if count <= 0:
        return True
    try:
        controller, Key, _ = _pynput_controller()
        for _ in range(count):
            controller.press(Key.backspace)
            controller.release(Key.backspace)
        return True
    except Exception:
        return False


def _pynput_type(text: str) -> bool:
    if not text:
        return True
    try:
        controller, _, _ = _pynput_controller()
        controller.type(text)
        return True
    except Exception:
        return False


def _xdotool_key(combo: str) -> bool:
    if not _has_xdotool():
        return False
    try:
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", combo],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _xdotool_backspace(count: int) -> bool:
    if count <= 0 or not _has_xdotool():
        return count <= 0
    try:
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "--repeat", str(count), "BackSpace"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _xdotool_type(text: str) -> bool:
    if not text:
        return True
    if not _has_xdotool():
        return False
    try:
        subprocess.run(
            ["xdotool", "type", "--delay", "1", "--", text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _to_platform_combo(combo: str) -> str:
    mapped = (combo or "").strip().lower()
    if sys.platform == "darwin":
        mapped = mapped.replace("ctrl+", "command+")
    return mapped


def _osascript_key(combo: str) -> bool:
    if sys.platform != "darwin":
        return False

    combo = _to_platform_combo(combo)
    parts = [part.strip() for part in combo.split("+") if part.strip()]
    if not parts:
        return False

    key = parts[-1]
    modifiers = parts[:-1]
    modifier_map = {
        "command": "command down",
        "cmd": "command down",
        "shift": "shift down",
        "option": "option down",
        "alt": "option down",
        "control": "control down",
        "ctrl": "control down",
    }
    valid_mods = [modifier_map[m] for m in modifiers if m in modifier_map]

    key_code_map = {
        "backspace": 51,
        "delete": 117,
        "return": 36,
        "enter": 36,
        "tab": 48,
        "escape": 53,
        "esc": 53,
    }

    try:
        if key in key_code_map:
            if valid_mods:
                script = (
                    f'tell application "System Events" to key code {key_code_map[key]} '
                    f"using {{{', '.join(valid_mods)}}}"
                )
            else:
                script = f'tell application "System Events" to key code {key_code_map[key]}'
        elif len(key) == 1:
            key_escaped = key.replace("\\", "\\\\").replace('"', '\\"')
            if valid_mods:
                script = (
                    f'tell application "System Events" to keystroke "{key_escaped}" '
                    f"using {{{', '.join(valid_mods)}}}"
                )
            else:
                script = f'tell application "System Events" to keystroke "{key_escaped}"'
        else:
            return False

        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _osascript_backspace(count: int) -> bool:
    if sys.platform != "darwin":
        return False
    if count <= 0:
        return True
    script = (
        'tell application "System Events"\n'
        f"repeat {count} times\n"
        "key code 51\n"
        "end repeat\n"
        "end tell"
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _osascript_type(text: str) -> bool:
    if sys.platform != "darwin":
        return False
    if not text:
        return True
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    script = f'tell application "System Events" to keystroke "{escaped}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


_X11_SENDER = None
_X11_INIT_FAILED = False


def _x11_sender():
    global _X11_SENDER, _X11_INIT_FAILED
    if _X11_INIT_FAILED:
        return None
    if _X11_SENDER is not None:
        return _X11_SENDER
    try:
        from x11_backend import X11Controller

        _X11_SENDER = X11Controller()
        return _X11_SENDER
    except Exception:
        _X11_INIT_FAILED = True
        return None


def _x11_key(combo: str) -> bool:
    sender = _x11_sender()
    if sender is None:
        return False
    try:
        return sender.combo(combo)
    except Exception:
        return False


def _x11_backspace(count: int) -> bool:
    if count <= 0:
        return True
    sender = _x11_sender()
    if sender is None:
        return False
    try:
        return sender.backspace(count)
    except Exception:
        return False


def _x11_type(text: str) -> bool:
    if not text:
        return True
    sender = _x11_sender()
    if sender is None:
        return False
    try:
        return sender.type_text(text)
    except Exception:
        return False


def _backend_order(selected: str) -> List[str]:
    if selected != "auto":
        return [selected]
    if sys.platform.startswith("linux"):
        return ["x11", "pynput", "xdotool", "keyboard"]
    if sys.platform == "darwin":
        return ["pynput", "osascript", "keyboard"]
    return ["keyboard", "pynput"]


def _send_hotkey(combo: str, backend: str = "auto") -> None:
    selected = (backend or "auto").strip().lower()
    combo = _to_platform_combo(combo)
    for candidate in _backend_order(selected):
        if candidate == "x11" and _x11_key(combo):
            return
        if candidate == "pynput" and _pynput_key(combo):
            return
        if candidate == "osascript" and _osascript_key(combo):
            return
        if candidate == "xdotool" and _xdotool_key(combo):
            return
        if candidate == "keyboard":
            try:
                keyboard = _keyboard_module()
                keyboard.press_and_release(combo)
                return
            except Exception:
                continue
    raise RuntimeError(
        f"Failed to send hotkey '{combo}'. Try PASTE_BACKEND=x11/pynput/xdotool/osascript and check input permissions."
    )


def _send_backspaces(count: int, backend: str = "auto") -> None:
    if count <= 0:
        return
    selected = (backend or "auto").strip().lower()
    for candidate in _backend_order(selected):
        if candidate == "x11" and _x11_backspace(count):
            return
        if candidate == "pynput" and _pynput_backspace(count):
            return
        if candidate == "osascript" and _osascript_backspace(count):
            return
        if candidate == "xdotool" and _xdotool_backspace(count):
            return
        if candidate == "keyboard":
            try:
                keyboard = _keyboard_module()
                for _ in range(count):
                    keyboard.press_and_release("backspace")
                return
            except Exception:
                continue
    raise RuntimeError(
        "Failed to send backspace events. Try PASTE_BACKEND=x11/pynput/xdotool/osascript and check input permissions."
    )


def _type_text(text: str, backend: str = "auto") -> None:
    if not text:
        return
    selected = (backend or "auto").strip().lower()
    for candidate in _backend_order(selected):
        if candidate == "x11" and _x11_type(text):
            return
        if candidate == "pynput" and _pynput_type(text):
            return
        if candidate == "osascript" and _osascript_type(text):
            return
        if candidate == "xdotool" and _xdotool_type(text):
            return
        if candidate == "keyboard":
            try:
                keyboard = _keyboard_module()
                keyboard.write(text)
                return
            except Exception:
                continue
    raise RuntimeError("Failed to type text. Try PASTE_BACKEND=x11/pynput/xdotool/osascript and check input permissions.")


def _copy_with_pyperclip(text: str) -> bool:
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:
        return False


def _paste_insert(text: str, backend: str = "auto") -> None:
    if not text:
        return

    if _copy_with_pyperclip(text):
        _send_hotkey("ctrl+v", backend=backend)
        return

    _type_text(text, backend=backend)


def paste_delta(prev_text: str, new_text: str, backend: str = "auto") -> None:
    to_delete, to_insert = compute_delta(prev_text, new_text)
    _send_backspaces(max(0, to_delete), backend=backend)
    _paste_insert(to_insert, backend=backend)


def paste_full_replace(new_text: str, backend: str = "auto") -> None:
    _send_hotkey("ctrl+a", backend=backend)
    _paste_insert(new_text, backend=backend)


def paste_text(prev_text: str, new_text: str, mode: str = "delta", backend: str = "auto") -> None:
    selected_mode = (mode or "delta").strip().lower()
    if selected_mode == "full":
        paste_full_replace(new_text or "", backend=backend)
        return
    paste_delta(prev_text or "", new_text or "", backend=backend)
