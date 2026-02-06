"""
DictaPilot - Cross-platform press-and-hold dictation with smart editing.

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
import threading
import tempfile
import queue
import time
import sys
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from dotenv import load_dotenv
from paste_utils import paste_text
from smart_editor import (
    TranscriptState,
    smart_update_state,
    llm_refine,
    sync_state_to_output,
    is_transform_command,
    DICTATION_MODE,
    CLEANUP_LEVEL,
)
from transcription_store import add_transcription, get_storage_info, export_all_to_text

# load environment variables from .env (if present)
load_dotenv()

try:
    from groq import Groq
except Exception:
    Groq = None

_GROQ_CLIENT = None

API_KEY = os.getenv("GROQ_API_KEY")
HOTKEY = os.getenv("HOTKEY", "f9")  # default hotkey: F9 (press-and-hold)
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo").strip() or "whisper-large-v3-turbo"
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


SMART_EDIT = _env_flag("SMART_EDIT", "1")
SMART_MODE = os.getenv("SMART_MODE", "llm").strip().lower()
PASTE_MODE = os.getenv("PASTE_MODE", "delta").strip().lower()
PASTE_BACKEND = os.getenv("PASTE_BACKEND", "auto").strip().lower()
HOTKEY_BACKEND = os.getenv("HOTKEY_BACKEND", "auto").strip().lower()
RESET_TRANSCRIPT_EACH_RECORDING = _env_flag("RESET_TRANSCRIPT_EACH_RECORDING", "1")
LLM_ALWAYS_CLEAN = _env_flag("LLM_ALWAYS_CLEAN", "1")
INSTANT_REFINE = _env_flag("INSTANT_REFINE", "1")
if SMART_MODE not in {"heuristic", "llm"}:
    SMART_MODE = "llm"
if PASTE_MODE not in {"delta", "full"}:
    PASTE_MODE = "delta"
if PASTE_BACKEND not in {"auto", "keyboard", "pynput", "xdotool", "x11", "osascript"}:
    PASTE_BACKEND = "auto"
if HOTKEY_BACKEND not in {"auto", "keyboard", "pynput", "x11"}:
    HOTKEY_BACKEND = "auto"

# audio defaults
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


SR = _env_int("SAMPLE_RATE", 16000)
CHANNELS = _env_int("CHANNELS", 1)
TRIM_SILENCE = _env_flag("TRIM_SILENCE", "1")
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.02"))


def _hotkey_token_for_pynput(hotkey: str, pynput_keyboard):
    key_name = (hotkey or "").strip().lower()
    mapping = {
        "ctrl": pynput_keyboard.Key.ctrl,
        "control": pynput_keyboard.Key.ctrl,
        "alt": pynput_keyboard.Key.alt,
        "shift": pynput_keyboard.Key.shift,
        "tab": pynput_keyboard.Key.tab,
        "enter": pynput_keyboard.Key.enter,
        "return": pynput_keyboard.Key.enter,
        "esc": pynput_keyboard.Key.esc,
        "escape": pynput_keyboard.Key.esc,
        "space": pynput_keyboard.Key.space,
    }
    if key_name in mapping:
        return mapping[key_name]
    if key_name.startswith("f") and key_name[1:].isdigit() and hasattr(pynput_keyboard.Key, key_name):
        return getattr(pynput_keyboard.Key, key_name)
    if len(key_name) == 1:
        return pynput_keyboard.KeyCode.from_char(key_name)
    return None


class HotkeyManager:
    def __init__(self, hotkey: str, on_press, on_release, backend: str = "auto"):
        self.hotkey = (hotkey or "").strip()
        self.on_press = on_press
        self.on_release = on_release
        self.backend = (backend or "auto").strip().lower()
        self.active_backend = None
        self._stop = None
        self._pressed = False

    def _try_start_keyboard(self):
        import keyboard

        press_hook = keyboard.on_press_key(self.hotkey, lambda _: self._handle_press())
        release_hook = keyboard.on_release_key(self.hotkey, lambda _: self._handle_release())

        def _stop():
            try:
                keyboard.unhook(press_hook)
            except Exception:
                pass
            try:
                keyboard.unhook(release_hook)
            except Exception:
                pass

        self._stop = _stop

    def _try_start_pynput(self):
        from pynput import keyboard as pynput_keyboard

        token = _hotkey_token_for_pynput(self.hotkey, pynput_keyboard)
        if token is None:
            raise ValueError(f"Unsupported hotkey for pynput backend: '{self.hotkey}'")

        def _matches(key_obj):
            try:
                if isinstance(token, pynput_keyboard.KeyCode):
                    return getattr(key_obj, "char", None) and key_obj.char.lower() == token.char.lower()
                return key_obj == token
            except Exception:
                return False

        def _on_press(key_obj):
            if _matches(key_obj):
                self._handle_press()

        def _on_release(key_obj):
            if _matches(key_obj):
                self._handle_release()

        listener = pynput_keyboard.Listener(on_press=_on_press, on_release=_on_release)
        listener.daemon = True
        listener.start()

        def _stop():
            try:
                listener.stop()
            except Exception:
                pass

        self._stop = _stop

    def _try_start_x11(self):
        from x11_backend import X11HotkeyListener

        listener = X11HotkeyListener(
            hotkey=self.hotkey,
            on_press=lambda: self._handle_press(),
            on_release=lambda: self._handle_release(),
        )
        listener.start()

        def _stop():
            try:
                listener.stop()
            except Exception:
                pass

        self._stop = _stop

    def _handle_press(self):
        if self._pressed:
            return
        self._pressed = True
        self.on_press(None)

    def _handle_release(self):
        if not self._pressed:
            return
        self._pressed = False
        self.on_release(None)

    def start(self):
        order = []
        if self.backend == "auto":
            # Linux avoids keyboard backend by default due /dev/input instability on many systems.
            if sys.platform.startswith("linux"):
                order = ["x11", "pynput"]
            elif sys.platform == "darwin":
                order = ["pynput", "keyboard"]
            else:
                order = ["keyboard", "pynput"]
        else:
            order = [self.backend]

        errors = []
        for candidate in order:
            try:
                if candidate == "keyboard":
                    self._try_start_keyboard()
                elif candidate == "pynput":
                    self._try_start_pynput()
                elif candidate == "x11":
                    self._try_start_x11()
                else:
                    raise ValueError(f"Unsupported hotkey backend '{candidate}'")
                self.active_backend = candidate
                return candidate
            except Exception as ex:
                errors.append(f"{candidate}: {ex}")

        raise RuntimeError("Unable to start hotkey listener. " + " | ".join(errors))

    def stop(self):
        if self._stop is not None:
            self._stop()
            self._stop = None


class Recorder:
    def __init__(self, samplerate=SR, channels=CHANNELS):
        self.sr = samplerate
        self.channels = channels
        self._active_sr = samplerate
        self._frames = []
        self._rec_thread = None
        self._running = threading.Event()
        self.last_error = None
        self._started_event = threading.Event()
        self.amplitude_callback = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Recording status:", status, file=sys.stderr)
        # copy because indata is reused by sounddevice
        self._frames.append(indata.copy())
        
        if self.amplitude_callback:
            # Calculate RMS amplitude for visualization
            try:
                rms = np.sqrt(np.mean(indata**2))
                self.amplitude_callback(float(rms))
            except Exception:
                pass

    def start(self, amplitude_callback=None):
        self._frames = []
        self.last_error = None
        self._started_event.clear()
        self._running.set()
        self.amplitude_callback = amplitude_callback
        def _run():
            errors = []
            candidates = []
            for candidate in (self.sr, 16000, 44100, 48000):
                if candidate not in candidates:
                    candidates.append(candidate)

            for candidate in candidates:
                try:
                    with sd.InputStream(samplerate=candidate, channels=self.channels, callback=self._callback):
                        self._active_sr = candidate
                        # signal that the input stream opened successfully
                        self._started_event.set()
                        while self._running.is_set():
                            sd.sleep(100)
                        return
                except Exception as e:
                    errors.append(f"{candidate}Hz: {e}")
                    self.last_error = str(e)
                    self._frames = []
                    continue

            if errors:
                self.last_error = " | ".join(errors)
            self._running.clear()

        self._rec_thread = threading.Thread(target=_run, daemon=True)
        self._rec_thread.start()

    def stop(self, outpath: str):
        self._running.clear()
        if self._rec_thread is not None:
            self._rec_thread.join()
        if self.last_error:
            raise RuntimeError(self.last_error)
        if not self._frames:
            raise RuntimeError("No audio recorded")
        data = np.concatenate(self._frames, axis=0)
        if TRIM_SILENCE:
            data = _trim_silence(data, SILENCE_THRESHOLD)
            if data.size == 0:
                raise RuntimeError("No audio detected after trimming")
        # ensure shape (n, channels)
        sf.write(outpath, data, self._active_sr)
        return outpath


def transcribe_with_groq(audio_path: str):
    if Groq is None:
        raise RuntimeError("Groq package not installed or failed to import")
    if not API_KEY:
        raise RuntimeError("Set GROQ_API_KEY environment variable first")

    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        _GROQ_CLIENT = Groq(api_key=API_KEY)
    client = _GROQ_CLIENT
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    resp = client.audio.transcriptions.create(
        file=(os.path.basename(audio_path), audio_bytes),
        model=GROQ_WHISPER_MODEL,
        temperature=0,
        response_format="verbose_json",
    )
    # object shape depends on SDK; try common access patterns
    if hasattr(resp, "text"):
        return resp.text
    if isinstance(resp, dict):
        return resp.get("text") or resp.get("transcription") or str(resp)
    return str(resp)


def _trim_silence(data: np.ndarray, threshold: float) -> np.ndarray:
    if data.size == 0:
        return data
    mono = np.abs(data.mean(axis=1)) if data.ndim > 1 else np.abs(data)
    peak = float(mono.max())
    if peak <= 0:
        return data
    cutoff = peak * max(threshold, 0.0)
    idx = np.where(mono > cutoff)[0]
    if idx.size == 0:
        return data[:0]
    return data[idx[0] : idx[-1] + 1]


class GUIManager:
    def __init__(self):
        self._queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self.root = tk.Tk()
        # Ensure window is persistent and always on top
        try:
            self.root.overrideredirect(True)
        except Exception:
            pass
        self.root.wm_attributes("-topmost", True)
        
        self._mode = "idle"
        self._display_text = "Ready"
        self._canvas = None
        self._font_header = tkfont.Font(family=None, size=9, weight="bold")
        self._font_content = tkfont.Font(family=None, size=10, weight="normal")
        self._amplitudes = [0.0] * 6  
        self._current_heights = [0.0] * 6  
        
        # Window Dragging State
        self._drag_data = {"x": 0, "y": 0}
        self._btn_hover = None
        
        # Initial window setup
        self._setup_window()
        self._poll()

    def _setup_window(self):
        transparent_key = "#123456"
        bg_color = "#0f172a" 
        
        try:
            self.root.configure(bg=transparent_key)
            self.root.wm_attributes("-transparentcolor", transparent_key)
        except Exception:
            self.root.configure(bg=bg_color)

        width = 260
        height = 140 
        if self._canvas:
            self._canvas.destroy()
            
        self._canvas = tk.Canvas(self.root, width=width, height=height, highlightthickness=0, bg=transparent_key)
        self._canvas.pack()
        
        # Bind events
        self._canvas.bind("<Button-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_motion)
        self._canvas.bind("<Motion>", self._on_mouse_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_click_release)
        self.root.bind("<Map>", self._on_window_map)
        
        # Position TOP-center
        self.root.update_idletasks()
        ws = self.root.winfo_screenwidth()
        x = (ws // 2) - (width // 2)
        y = 30
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        self._render_frame()

    def _on_window_map(self, event):
        # Restore borderless state when restored from taskbar
        try:
            self.root.overrideredirect(True)
        except Exception:
            pass

    def _on_drag_start(self, event):
        btn = self._get_button_at(event.x, event.y)
        if btn: return
            
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
        self._canvas.config(cursor="fleur")

    def _on_drag_motion(self, event):
        if self._canvas.cget("cursor") != "fleur": return
        deltax = event.x - self._drag_data["x"]
        deltay = event.y - self._drag_data["y"]
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    def _on_mouse_move(self, event):
        btn = self._get_button_at(event.x, event.y)
        if btn != self._btn_hover:
            self._btn_hover = btn
            self._render_frame()
            
        if btn: self._canvas.config(cursor="hand2")
        else: self._canvas.config(cursor="")

    def _on_click_release(self, event):
        self._canvas.config(cursor="")
        btn = self._get_button_at(event.x, event.y)
        if btn == "close": self.root.quit()
        elif btn == "min":
            self.root.overrideredirect(False)
            self.root.iconify()

    def _get_button_at(self, x, y):
        # Coordinates based on width=260
        if 235 <= x <= 255 and 5 <= y <= 25: return "close"
        if 210 <= x <= 230 and 5 <= y <= 25: return "min"
        return None

    def _poll(self):
        try:
            while True:
                cmd, arg = self._queue.get_nowait()
                if cmd == "show" or cmd == "update":
                    mode = "textpopup"
                    body = arg
                    if isinstance(arg, (tuple, list)):
                        mode, body = arg[0], arg[1]
                    self._mode = mode
                    self._display_text = body
                elif cmd == "close":
                    self._mode = "idle"
                    self._display_text = "Ready"
        except queue.Empty:
            pass

        # Process audio levels
        try:
            while True:
                amp = self._audio_queue.get_nowait()
                self._amplitudes.pop(0)
                self._amplitudes.append(amp)
        except queue.Empty:
            pass
        
        # Animation Tick
        decay = 0.12 
        rise = 0.6   
        sensitivities = [0.6, 1.1, 1.6, 1.6, 1.1, 0.6]
        
        for i in range(6):
            if self._mode == "record":
                target = self._amplitudes[i] * sensitivities[i] * 90.0
            elif self._mode == "idle":
                import math
                target = (math.sin(time.time() * 2 + i) + 1) * 0.04
            else:
                target = 0.05 
                
            diff = target - self._current_heights[i]
            if diff > 0:
                self._current_heights[i] += diff * rise
            else:
                self._current_heights[i] += diff * decay

        self._render_frame()
        self.root.after(20, self._poll)

    def _draw_rounded_rect(self, x, y, w, h, color, r=15):
        self._canvas.create_rectangle(x+r, y, x+w-r, y+h, fill=color, outline=color)
        self._canvas.create_rectangle(x, y+r, x+w, y+h-r, fill=color, outline=color)
        self._canvas.create_oval(x, y, x+2*r, y+2*r, fill=color, outline=color)
        self._canvas.create_oval(x+w-2*r, y, x+w, y+2*r, fill=color, outline=color)
        self._canvas.create_oval(x, y+h-2*r, x+2*r, y+h, fill=color, outline=color)
        self._canvas.create_oval(x+w-2*r, y+h-2*r, x+w, y+h, fill=color, outline=color)

    def _render_frame(self):
        if not self._canvas: return
        self._canvas.delete("all")
        
        w = int(self._canvas.cget("width"))
        h = int(self._canvas.cget("height"))
        
        # Background
        self._draw_rounded_rect(0, 0, w, h, "#0f172a") 
        
        # Action Buttons
        close_color = "#ef4444" if self._btn_hover == "close" else "#475569"
        min_color = "#38bdf8" if self._btn_hover == "min" else "#475569"
        self._canvas.create_text(245, 15, text="✕", fill=close_color, font=("Helvetica", 10, "bold"))
        self._canvas.create_text(220, 15, text="—", fill=min_color, font=("Helvetica", 10, "bold"))

        # Zone A: Branding
        status_map = {"record": ("RECORDING", "#38bdf8"), "processing": ("THINKING...", "#fbbf24"), "done": ("COMPLETED", "#4ade80"), "idle": ("IDLE", "#94a3b8")}
        status_text, status_color = status_map.get(self._mode, ("READY", "#94a3b8"))
        self._canvas.create_text(w//2, 20, text="DICTAPILOT", fill="#f1f5f9", font=self._font_header)
        self._canvas.create_text(w//2, 35, text=status_text, fill=status_color, font=("Helvetica", 7, "bold"))

        # Zone B: Visualizer
        mid_y = 75
        bar_w, gap, max_h = 16, 12, 45
        total_w = 6 * (bar_w + gap) - gap
        start_x = (w - total_w) // 2
        visual_color = "#38bdf8" if self._mode == "record" else "#1e293b"
        glow_color = "#0c4a6e" if self._mode == "record" else "#0f172a"
        for i in range(6):
            hv = max(4, int(min(1.0, self._current_heights[i]) * max_h))
            x = start_x + i * (bar_w + gap)
            self._canvas.create_line(x+bar_w/2, mid_y-hv/2-2, x+bar_w/2, mid_y+hv/2+2, fill=glow_color, width=bar_w+4, capstyle="round")
            self._canvas.create_line(x+bar_w/2, mid_y-hv/2, x+bar_w/2, mid_y+hv/2, fill=visual_color, width=bar_w, capstyle="round")

        # Zone C: Text
        self._draw_rounded_rect(10, 100, w-20, 30, "#1e293b", r=8)
        msg = self._display_text
        if self._mode == "record": msg = "Listening..."
        elif self._mode == "processing": msg = "Analyzing audio..."
        if len(msg) > 35: msg = msg[:32] + "..."
        self._canvas.create_text(w//2, 115, text=msg, fill="#cbd5e1", font=self._font_content)

    def show(self, text: str):
        self._queue.put(("show", text))

    def update(self, text: str):
        self._queue.put(("update", text))

    def update_amplitude(self, amp: float):
        self._audio_queue.put(amp)

    def close(self):
        self._queue.put(("close", None))


def main():
    def print_banner(hotkey: str):
        banner = r"""
 ____  _ _      _        ____  _ _       _ _   
|  _ \(_) |    | |      |  _ \(_) |     | | |  
| | | |_| | ___| |_ __ _| |_) || | ___  | | |_ 
| | | | | |/ __| __/ _` |  ___/ | |/ _ \ | | __|
| |_| | | | (__| || (_| | |   | | | (_) || | |_ 
|____/|_|_|\___|\__\__,_|_|   |_|_|\___/ |_|\__|
"""
        print(banner)
        print("DictaPilot")
        print("Developer: Rehan")
        print("License: MIT (see LICENSE file)")
        print("")
        print(f"Hold '{hotkey}' to record; release to send audio for transcription.")
        print(
            f"Smart dictation: {'on' if SMART_EDIT else 'off'} "
            f"(mode={SMART_MODE}, paste={PASTE_MODE}, paste_backend={PASTE_BACKEND})"
        )
        if SMART_EDIT and SMART_MODE == "llm":
            cleanup_mode = "always" if LLM_ALWAYS_CLEAN else "intent-only"
            print(f"LLM cleanup: {cleanup_mode} (chat_model={GROQ_CHAT_MODEL})")
        print(f"Transcription model: {GROQ_WHISPER_MODEL}")
        print(f"Hotkey backend preference: {HOTKEY_BACKEND}")
        print(f"Dictation mode: {DICTATION_MODE} (cleanup={CLEANUP_LEVEL})")
        print(f"Audio: {SR}Hz, channels={CHANNELS}, trim_silence={'on' if TRIM_SILENCE else 'off'}")
        print(f"Instant refine: {'on' if INSTANT_REFINE else 'off'}")
        if SMART_EDIT and RESET_TRANSCRIPT_EACH_RECORDING:
            print("Transcript reset mode: per recording")
        elif SMART_EDIT:
            print("Transcript reset mode: session (keeps previous recordings)")

        try:
            storage_info = get_storage_info()
            print(f"Transcription storage: {storage_info['storage_path']}")
            print(f"Total transcriptions: {storage_info['statistics']['total_transcriptions']}")
        except Exception:
            pass
        print("")

    print_banner(HOTKEY)
    recorder = Recorder()

    gui = GUIManager()
    transcript_state = TranscriptState()

    def on_press(e):
        # start only if not already recording
        if recorder._running.is_set():
            return
        if SMART_EDIT and RESET_TRANSCRIPT_EACH_RECORDING:
            with transcript_state.lock:
                transcript_state.segments.clear()
                transcript_state.output_text = ""
        print("Start recording")
        try:
            gui.show(("record", "Recording..."))
        except Exception:
            pass
        recorder.last_error = None
        recorder.start(amplitude_callback=gui.update_amplitude)
        # wait for the input stream to open or error
        started = recorder._started_event.wait(timeout=1.0)
        if not started:
            # if there was an error, show it; otherwise show a timeout message
            msg = recorder.last_error or "Timeout opening audio input device"
            gui.update(f"Recording error: {msg}")
            time.sleep(1.5)
            gui.close()
            return

    def on_release(e):
        # ignore if we never started recording
        if not recorder._running.is_set():
            return
        print("Stop recording")
        try:
            # save to temp file
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            gui.update("Stopping... saving audio")
            audio_path = recorder.stop(path)
        except Exception as ex:
            print("Recording error:", ex, file=sys.stderr)
            gui.update(f"Recording error: {ex}")
            time.sleep(2)
            gui.close()
            return

        def _process():
            try:
                gui.update(("processing", "Processing audio..."))
                text = transcribe_with_groq(audio_path)
                if not text:
                    text = "(no transcription returned)"
                print("Transcription:\n", text)

                prev_out = transcript_state.output_text
                fast_out = text
                fast_action = "append"

                if SMART_EDIT:
                    prev_out, fast_out, fast_action = smart_update_state(
                        transcript_state,
                        text,
                        mode="heuristic",
                        allow_llm=False,
                    )
                    print(f"Fast action: {fast_action}")
                    print("Fast transcript:\n", fast_out if fast_out else "(empty)")

                # copy to clipboard (prefer pyperclip, fallback to tkinter clipboard)
                try:
                    import pyperclip
                    pyperclip.copy(fast_out)
                except Exception:
                    try:
                        # use the GUI root for clipboard operations (must be main thread)
                        def _cb():
                            try:
                                gui.root.clipboard_clear()
                                gui.root.clipboard_append(fast_out)
                                gui.root.update()
                            except Exception:
                                pass
                        gui.root.after(0, _cb)
                    except Exception:
                        pass

                # Close popup before paste so focus can return to the target app (e.g., Notepad).
                try:
                    gui.close()
                except Exception:
                    pass

                # small delay to allow focus handoff, then paste fast output
                time.sleep(0.25)
                paste_error = None
                try:
                    selected_paste_mode = PASTE_MODE if SMART_EDIT else "delta"
                    paste_text(prev_out, fast_out, selected_paste_mode, backend=PASTE_BACKEND)
                except Exception as ex:
                    paste_error = ex
                    try:
                        # fallback: force full replace for better compatibility
                        paste_text("", fast_out, "full", backend=PASTE_BACKEND)
                        paste_error = None
                    except Exception as ex2:
                        paste_error = RuntimeError(f"{ex}; fallback full paste failed: {ex2}")

                if paste_error is not None:
                    print(f"Paste error: {paste_error}", file=sys.stderr)
                    try:
                        gui.update("Paste failed. Try normal user mode and PASTE_BACKEND=x11/xdotool.")
                    except Exception:
                        pass

                # optional refinement pass (LLM) to improve quality
                refined_out = fast_out
                refined_action = fast_action
                if SMART_EDIT and INSTANT_REFINE and DICTATION_MODE != "speed" and not is_transform_command(text):
                    try:
                        gui.update(("processing", "Refining..."))
                    except Exception:
                        pass
                    llm_result = llm_refine(prev_out, text)
                    if llm_result is not None:
                        refined_out, refined_action = llm_result
                        if refined_out != fast_out:
                            with transcript_state.lock:
                                current_out = transcript_state.output_text
                            if current_out == fast_out:
                                sync_state_to_output(transcript_state, fast_out, refined_out)
                                try:
                                    paste_text(fast_out, refined_out, "delta", backend=PASTE_BACKEND)
                                except Exception:
                                    try:
                                        paste_text("", refined_out, "full", backend=PASTE_BACKEND)
                                    except Exception:
                                        pass

                # Save transcription to storage (final output)
                add_transcription(text, refined_out, refined_action)
                print("Transcription saved to storage")

                # show done text briefly after paste
                try:
                    output_for_popup = refined_out if refined_out else "(empty transcript)"
                    snippet = output_for_popup if len(output_for_popup) <= 300 else output_for_popup[:300] + "..."
                    gui.show(("done", snippet))
                except Exception:
                    pass

                # keep the done window visible briefly then return to idle
                time.sleep(1.5)
                try:
                    gui.show(("idle", "Ready"))
                except Exception:
                    pass

            except Exception as ex:
                gui.update(f"Transcription error: {ex}")
                print("Transcription error:", ex, file=sys.stderr)
            finally:
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

        t = threading.Thread(target=_process, daemon=True)
        t.start()

    # register hotkey handlers
    hotkey_manager = HotkeyManager(HOTKEY, on_press, on_release, backend=HOTKEY_BACKEND)
    try:
        active_backend = hotkey_manager.start()
        print(f"Hotkey listener active: {active_backend}")
    except Exception as ex:
        print(f"Failed to register hotkey '{HOTKEY}': {ex}", file=sys.stderr)
        return

    print("Ready. Press and hold the hotkey to record.")
    try:
        gui.root.mainloop()
    except KeyboardInterrupt:
        print("Exiting")
        try:
            gui.root.quit()
        except Exception:
            pass
    finally:
        try:
            hotkey_manager.stop()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DictaPilot - Press-and-hold dictation")
    parser.add_argument("--tray", action="store_true", help="Run with system tray")
    parser.add_argument("--export", type=str, metavar="FILE", help="Export all transcriptions to a text file")
    parser.add_argument("--list", action="store_true", help="List recent transcriptions")
    parser.add_argument("--stats", action="store_true", help="Show transcription statistics")
    parser.add_argument("--search", type=str, metavar="QUERY", help="Search transcriptions")
    args = parser.parse_args()

    if args.export:
        from transcription_store import export_all_to_text
        path = Path(args.export)
        content = export_all_to_text(path, include_metadata=True)
        print(f"Exported to: {path}")
        print(f"Content preview:\n{content[:500]}...")
        sys.exit(0)

    if args.list:
        from transcription_store import get_transcriptions
        entries = get_transcriptions(20)
        print("Recent Transcriptions:")
        print("-" * 60)
        for i, entry in enumerate(entries, 1):
            print(f"{i}. [{entry.timestamp[:19]}] {entry.display_text[:80]}")
            if len(entry.display_text) > 80:
                print(f"   ... ({entry.word_count} words, action: {entry.action})")
        sys.exit(0)

    if args.stats:
        from transcription_store import get_storage_info
        info = get_storage_info()
        print("Transcription Storage Statistics")
        print("-" * 40)
        print(f"Storage location: {info['storage_path']}")
        stats = info['statistics']
        print(f"Total transcriptions: {stats['total_transcriptions']}")
        print(f"Total words: {stats['total_words']}")
        print(f"Total characters: {stats['total_characters']}")
        print(f"Action breakdown: {stats['action_breakdown']}")
        sys.exit(0)

    if args.search:
        from transcription_store import search_transcriptions
        results = search_transcriptions(args.search)
        print(f"Search results for '{args.search}':")
        print("-" * 60)
        for entry in results:
            print(f"[{entry.timestamp[:19]}] {entry.display_text[:100]}")
        print(f"Found {len(results)} matching transcriptions")
        sys.exit(0)

    main()
