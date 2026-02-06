# GEMINI.md

## Project Overview
**DictaPilot** is a cross-platform (Windows, macOS, Linux) press-and-hold dictation application. It enables users to record audio by holding a hotkey (default `F9`), transcribe it using **Groq Whisper**, and automatically paste the result into the focused application. It features "Smart Editing" which allows users to perform text manipulations via voice commands (e.g., "delete that", "clear all", "replace X with Y").

### Main Technologies
- **Language:** Python 3.10+
- **Transcription:** Groq API (Whisper-large-v3-turbo)
- **Smart Editing:** Groq API (LLM-based refinement) + Heuristics
- **Audio Handling:** `sounddevice`, `soundfile`, `numpy`
- **Input/Global Hotkeys:** `keyboard`, `pynput`, and platform-specific backends (X11 for Linux, osascript for macOS)
- **UI:** Tkinter (for status popups) and `pystray` (for system tray integration)
- **Persistence:** JSON-based storage for configurations, transcription history, and per-app profiles

### Architecture
- `app.py`: The central orchestrator handling the recording lifecycle, API communication, and GUI updates.
- `smart_editor.py`: Contains the logic for interpreting voice commands and refining transcripts using both regex-based heuristics and LLM calls.
- `paste_utils.py`: Manages cross-platform text injection, including "delta-pasting" which only sends the necessary keystrokes (backspaces/typing) to update the current text.
- `transcription_store.py`: Handles persistent logging of all transcriptions to a local JSON file.
- `app_context.py`: Detects the currently active window to apply app-specific profiles (e.g., formal tone for Gmail, casual for Slack).
- `config.py`: Manages application settings with support for environment variable overrides.

---

## Building and Running

### Prerequisites
- Python 3.10 or higher
- A valid **Groq API Key**

### Setup
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configuration:**
   Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
   ```bash
   cp .env.example .env
   ```

### Running the Application
- **Standard Mode:**
  ```bash
  python app.py
  ```
- **Tray Mode:**
  ```bash
  python app.py --tray
  ```

### CLI Tools
DictaPilot includes built-in tools for managing transcription history:
- **List History:** `python app.py --list`
- **View Stats:** `python app.py --stats`
- **Search:** `python app.py --search "query"`
- **Export:** `python app.py --export history.txt`

### Testing
Tests are located in the `tests/` directory and use `pytest`.
```bash
pytest
```

### Packaging
Platform-specific build scripts are available in the `packaging/` directory:
- Linux: `./packaging/build_linux.sh`
- macOS: `./packaging/build_macos.sh`
- Windows: `.\packaging\build_windows.ps1`

---

## Development Conventions

### Cross-Platform Backends
The project maintains high compatibility by implementing multiple backends for critical operations:
- **Hotkey Backends:** `keyboard`, `pynput`, `x11`.
- **Paste Backends:** `keyboard`, `pynput`, `xdotool`, `x11`, `osascript`.
Always check for platform-specific edge cases when modifying input/output logic.

### Smart Dictation Logic
Smart editing is split into two layers:
1. **Heuristics:** Fast, regex-based matching for common commands (e.g., "clear", "undo").
2. **LLM Refinement:** High-accuracy cleanup and complex command handling (e.g., "make it more formal").
Avoid introducing heavy dependencies in the heuristic layer to maintain low latency.

### Delta Pasting
Instead of replacing the entire text field, DictaPilot computes the difference between the previous and current transcript and sends only the required `BackSpace` and character events. Ensure any changes to `paste_utils.py` preserve this efficiency.

### Persistence
- **Config:** `config.json` in platform-specific app data.
- **Transcriptions:** `transcriptions.json` in platform-specific data folders.
- **Profiles:** `profiles.json` for per-app tone/language settings.

### Contribution Workflow
- Follow the existing style (clean Python with type hints where appropriate).
- Add unit tests for new smart editor commands in `tests/`.
- Ensure new features are configurable via environment variables in `config.py`.
