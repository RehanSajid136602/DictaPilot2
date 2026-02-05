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

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Recording status:", status, file=sys.stderr)
        # copy because indata is reused by sounddevice
        self._frames.append(indata.copy())

    def start(self):
        self._frames = []
        self.last_error = None
        self._started_event.clear()
        self._running.set()
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
        self.root = tk.Tk()
        # keep the root window hidden; use Toplevel windows for status
        self.root.withdraw()
        self._window = None
        self._label = None
        self._mode = "textpopup"
        self._canvas = None
        self._text_id = None
        self._font_obj = None
        self._poll()

    def _poll(self):
        try:
            while True:
                cmd, arg = self._queue.get_nowait()
                if cmd == "show":
                    self._do_show(arg)
                elif cmd == "update":
                    self._do_update(arg)
                elif cmd == "close":
                    self._do_close()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _do_show(self, text: str):
        # Simplified dark rounded text-only popup placed bottom-center.
        # Accept either a plain text or a (mode, text) tuple; we only use the text here.
        body = text[1] if (isinstance(text, (tuple, list)) and len(text) > 1) else text

        if self._window:
            self._do_update(body)
            return

        self._window = tk.Toplevel(self.root)
        try:
            self._window.overrideredirect(True)
        except Exception:
            pass
        self._window.wm_attributes("-topmost", True)

        # Create a unique transparent color key for the window background
        transparent_key = "#123456"
        bg_color = "#222222"
        fg_color = "#ffffff"
        pad_x = 18
        pad_y = 10

        # set the window background to the transparent key
        try:
            self._window.configure(bg=transparent_key)
        except Exception:
            pass

        # Canvas will draw the rounded rectangle and the text
        font_obj = tkfont.Font(family=None, size=11)
        text_width = font_obj.measure(body)
        text_height = font_obj.metrics("linespace")
        width = text_width + pad_x * 2
        height = text_height + pad_y * 2

        canvas = tk.Canvas(self._window, width=width, height=height, highlightthickness=0, bg=transparent_key)
        canvas.pack()

        # draw rounded rectangle using ovals and rectangles
        r = 12
        color = bg_color
        # center rectangle parts
        canvas.create_rectangle(r, 0, width - r, height, fill=color, outline=color)
        canvas.create_rectangle(0, r, width, height - r, fill=color, outline=color)
        # four corner ovals
        canvas.create_oval(0, 0, 2 * r, 2 * r, fill=color, outline=color)
        canvas.create_oval(width - 2 * r, 0, width, 2 * r, fill=color, outline=color)
        canvas.create_oval(0, height - 2 * r, 2 * r, height, fill=color, outline=color)
        canvas.create_oval(width - 2 * r, height - 2 * r, width, height, fill=color, outline=color)

        # draw text
        self._text_id = canvas.create_text(width // 2, height // 2, text=body, fill=fg_color, font=font_obj)

        # try to make the background transparent (Windows supports -transparentcolor)
        try:
            self._window.wm_attributes("-transparentcolor", transparent_key)
        except Exception:
            try:
                self._window.configure(bg=color)
            except Exception:
                pass

        # position bottom-center
        self._window.update_idletasks()
        ws = self._window.winfo_screenwidth()
        hs = self._window.winfo_screenheight()
        x = (ws // 2) - (width // 2)
        y = hs - height - 60
        self._window.geometry(f"{width}x{height}+{x}+{y}")
        self._mode = "textpopup"
        self._canvas = canvas
        self._font_obj = font_obj

    def _do_update(self, text: str):
        # accept either a plain text or a (mode, text) tuple
        mode = None
        body = text
        if isinstance(text, tuple) or isinstance(text, list):
            mode, body = text[0], text[1]

        if mode and mode != self._mode:
            # recreate window with new mode
            self._do_close()
            self._do_show((mode, body))
            return

        if not self._window or not self._canvas or not self._font_obj:
            self._do_show((mode, body) if mode else body)
            return

        body = str(body or "")
        text_width = self._font_obj.measure(body)
        text_height = self._font_obj.metrics("linespace")
        pad_x = 18
        pad_y = 10
        width = text_width + pad_x * 2
        height = text_height + pad_y * 2
        self._canvas.config(width=width, height=height)

        # redraw rounded rect and text with the new dimensions
        self._canvas.delete("all")
        r = 12
        color = "#222222"
        self._canvas.create_rectangle(r, 0, width - r, height, fill=color, outline=color)
        self._canvas.create_rectangle(0, r, width, height - r, fill=color, outline=color)
        self._canvas.create_oval(0, 0, 2 * r, 2 * r, fill=color, outline=color)
        self._canvas.create_oval(width - 2 * r, 0, width, 2 * r, fill=color, outline=color)
        self._canvas.create_oval(0, height - 2 * r, 2 * r, height, fill=color, outline=color)
        self._canvas.create_oval(width - 2 * r, height - 2 * r, width, height, fill=color, outline=color)
        self._text_id = self._canvas.create_text(
            width // 2, height // 2, text=body, fill="#ffffff", font=self._font_obj
        )

        self._window.update_idletasks()
        ws = self._window.winfo_screenwidth()
        hs = self._window.winfo_screenheight()
        x = (ws // 2) - (width // 2)
        y = hs - height - 60
        self._window.geometry(f"{width}x{height}+{x}+{y}")

    def _do_close(self):
        if self._window:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None
            self._label = None

    def show(self, text: str):
        self._queue.put(("show", text))

    def update(self, text: str):
        self._queue.put(("update", text))

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
        recorder.start()
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

                # keep the done window visible briefly then close
                time.sleep(1.2)
                try:
                    gui.close()
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
