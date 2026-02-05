"""
DictaPilot X11 Backend
Linux X11 input backend for hotkey detection and text injection.

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

import ctypes
import ctypes.util
import os
import threading
import time
from typing import Callable, Optional


class X11UnavailableError(RuntimeError):
    pass


def _normalize_key_name(key_name: str) -> str:
    key = (key_name or "").strip().lower()
    mapping = {
        "ctrl": "Control_L",
        "control": "Control_L",
        "alt": "Alt_L",
        "shift": "Shift_L",
        "enter": "Return",
        "return": "Return",
        "esc": "Escape",
        "escape": "Escape",
        "space": "space",
        "tab": "Tab",
        "backspace": "BackSpace",
        "delete": "Delete",
    }
    if key in mapping:
        return mapping[key]
    if key.startswith("f") and key[1:].isdigit():
        return f"F{key[1:]}"
    if len(key) == 1:
        return key
    return key_name


class X11Controller:
    def __init__(self):
        if not os.getenv("DISPLAY"):
            raise X11UnavailableError("DISPLAY is not set")

        x11_path = ctypes.util.find_library("X11")
        xtst_path = ctypes.util.find_library("Xtst")
        if not x11_path or not xtst_path:
            raise X11UnavailableError("X11/Xtst libraries not found")

        self._x11 = ctypes.cdll.LoadLibrary(x11_path)
        self._xtst = ctypes.cdll.LoadLibrary(xtst_path)

        self._x11.XInitThreads.restype = ctypes.c_int
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._x11.XCloseDisplay.restype = ctypes.c_int
        self._x11.XFlush.argtypes = [ctypes.c_void_p]
        self._x11.XFlush.restype = ctypes.c_int
        self._x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
        self._x11.XStringToKeysym.restype = ctypes.c_ulong
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XKeysymToKeycode.restype = ctypes.c_uint
        self._x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._x11.XQueryKeymap.restype = ctypes.c_int

        self._xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
        self._xtst.XTestFakeKeyEvent.restype = ctypes.c_int

        self._x11.XInitThreads()
        self._display = self._x11.XOpenDisplay(None)
        if not self._display:
            raise X11UnavailableError("Unable to open X display")

        self._keymap_state = ctypes.create_string_buffer(32)
        self._lock = threading.Lock()

    def close(self) -> None:
        if self._display:
            self._x11.XCloseDisplay(self._display)
            self._display = None

    def _keysym_to_keycode(self, key_name: str) -> int:
        normalized = _normalize_key_name(key_name)
        if not normalized:
            return 0
        keysym = self._x11.XStringToKeysym(normalized.encode("utf-8"))
        if not keysym:
            return 0
        keycode = int(self._x11.XKeysymToKeycode(self._display, ctypes.c_ulong(keysym)))
        return keycode

    def keycode_for(self, key_name: str) -> int:
        with self._lock:
            return self._keysym_to_keycode(key_name)

    def is_pressed(self, key_name: str) -> bool:
        with self._lock:
            keycode = self._keysym_to_keycode(key_name)
            if keycode <= 0:
                return False
            self._x11.XQueryKeymap(self._display, self._keymap_state)
            byte_val = self._keymap_state.raw[keycode >> 3]
            mask = 1 << (keycode & 7)
            return (byte_val & mask) != 0

    def _fake_keycode(self, keycode: int, is_press: bool) -> bool:
        if keycode <= 0:
            return False
        rc = self._xtst.XTestFakeKeyEvent(self._display, ctypes.c_uint(keycode), 1 if is_press else 0, 0)
        return bool(rc)

    def tap(self, key_name: str) -> bool:
        with self._lock:
            keycode = self._keysym_to_keycode(key_name)
            if keycode <= 0:
                return False
            ok = self._fake_keycode(keycode, True) and self._fake_keycode(keycode, False)
            self._x11.XFlush(self._display)
            return ok

    def backspace(self, count: int) -> bool:
        if count <= 0:
            return True
        with self._lock:
            keycode = self._keysym_to_keycode("BackSpace")
            if keycode <= 0:
                return False
            for _ in range(count):
                if not self._fake_keycode(keycode, True):
                    return False
                if not self._fake_keycode(keycode, False):
                    return False
            self._x11.XFlush(self._display)
            return True

    def combo(self, combo: str) -> bool:
        keys = [part.strip() for part in (combo or "").split("+") if part.strip()]
        if not keys:
            return False

        with self._lock:
            keycodes = [self._keysym_to_keycode(key) for key in keys]
            if any(code <= 0 for code in keycodes):
                return False

            for code in keycodes[:-1]:
                if not self._fake_keycode(code, True):
                    return False

            if not self._fake_keycode(keycodes[-1], True):
                return False
            if not self._fake_keycode(keycodes[-1], False):
                return False

            for code in reversed(keycodes[:-1]):
                if not self._fake_keycode(code, False):
                    return False

            self._x11.XFlush(self._display)
            return True

    def type_text(self, text: str) -> bool:
        if not text:
            return True

        shift_required = {
            "~": "grave",
            "!": "1",
            "@": "2",
            "#": "3",
            "$": "4",
            "%": "5",
            "^": "6",
            "&": "7",
            "*": "8",
            "(": "9",
            ")": "0",
            "_": "minus",
            "+": "equal",
            "{": "bracketleft",
            "}": "bracketright",
            "|": "backslash",
            ":": "semicolon",
            '"': "apostrophe",
            "<": "comma",
            ">": "period",
            "?": "slash",
        }

        direct_map = {
            "\n": "Return",
            "\t": "Tab",
            " ": "space",
            "-": "minus",
            "=": "equal",
            "[": "bracketleft",
            "]": "bracketright",
            "\\": "backslash",
            ";": "semicolon",
            "'": "apostrophe",
            ",": "comma",
            ".": "period",
            "/": "slash",
            "`": "grave",
        }

        with self._lock:
            shift_code = self._keysym_to_keycode("Shift_L")
            if shift_code <= 0:
                return False

            for ch in text:
                use_shift = False
                key_name = None

                if ch in shift_required:
                    key_name = shift_required[ch]
                    use_shift = True
                elif ch in direct_map:
                    key_name = direct_map[ch]
                elif ch.isalpha():
                    key_name = ch.lower()
                    use_shift = ch.isupper()
                elif ch.isdigit():
                    key_name = ch
                else:
                    key_name = ch

                keycode = self._keysym_to_keycode(key_name)
                if keycode <= 0:
                    return False

                if use_shift and not self._fake_keycode(shift_code, True):
                    return False
                if not self._fake_keycode(keycode, True):
                    return False
                if not self._fake_keycode(keycode, False):
                    return False
                if use_shift and not self._fake_keycode(shift_code, False):
                    return False

            self._x11.XFlush(self._display)
            return True


class X11HotkeyListener:
    def __init__(
        self,
        hotkey: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        poll_interval: float = 0.02,
    ):
        self.hotkey = hotkey
        self.on_press = on_press
        self.on_release = on_release
        self.poll_interval = poll_interval
        self._controller: Optional[X11Controller] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pressed = False

    def start(self) -> None:
        self._controller = X11Controller()
        keycode = self._controller.keycode_for(self.hotkey)
        if keycode <= 0:
            self._controller.close()
            self._controller = None
            raise X11UnavailableError(f"Unsupported hotkey '{self.hotkey}' for X11 backend")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._controller is not None
        while not self._stop_event.is_set():
            try:
                is_down = self._controller.is_pressed(self.hotkey)
            except Exception:
                break

            if is_down and not self._pressed:
                self._pressed = True
                try:
                    self.on_press()
                except Exception:
                    pass
            elif not is_down and self._pressed:
                self._pressed = False
                try:
                    self.on_release()
                except Exception:
                    pass

            time.sleep(self.poll_interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        if self._controller is not None:
            try:
                self._controller.close()
            except Exception:
                pass
            self._controller = None
