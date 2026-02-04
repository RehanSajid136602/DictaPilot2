# DictaPilot

![DictaPilot](Dictepilot.png)

DictaPilot is a cross-platform press-and-hold dictation app. Hold a hotkey to record, transcribe with Groq Whisper, and paste directly into your focused text field with smart dictation edits.

**Part of the BridgeMind Vibeathon**
**Developer**: Rehan

**Project files**: [app.py](app.py), [requirements.txt](requirements.txt), [.env.example](.env.example)

**Quick summary**
- **App model used (Groq)**: `whisper-large-v3-turbo`
- **Required env var**: `GROQ_API_KEY`
- **Hotkey**: configurable via `HOTKEY` (default `f9`)
- **Smart dictation**: on by default (`SMART_EDIT=1`)
- **Cleanup mode**: LLM-first by default (`SMART_MODE=llm`, `LLM_ALWAYS_CLEAN=1`)
- **Text polish**: removes filler/repetition and improves punctuation automatically

Getting started (macOS / Linux / Windows)

1) Clone the repo and open a terminal in the project folder.

2) Create and activate a Python virtual environment

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Create a `.env` file from `.env.example` and add your GROQ_API_KEY
```bash
cp .env.example .env    # macOS / Linux
# or copy the file in Explorer on Windows, then edit
```
Edit `.env` and set `GROQ_API_KEY` to your Groq API key.

5) Run the app
```bash
python app.py
```
Then press-and-hold the configured hotkey (default `f9`) to record; release to send the audio to Groq and paste the transcription into the focused app.

Smart dictation commands
- `delete that`, `delete previous`, `undo`, `scratch that`, `remove that`, `remove previous`, `erase that`, `take that out`: undo the last committed segment.
- `clear all`, `clear everything`, `reset`, `start over`: clear the whole transcript state.
- `don't include that`, `do not include`, `don't add that`, `ignore that`, `ignore it`, `skip that`, `disregard that`, `omit that`, `never mind`: ignore this utterance and keep current transcript unchanged.
- `delete that ... <new content>`: undo last segment, then append the remainder.
- Inline self-correction in one utterance is handled conservatively (example: `My name is Rehan. No, no, my name is Numan` keeps only the corrected clause).

Examples
- Say: `Hello world.` -> transcript appends as normal.
- Say: `oh no delete that` -> removes the previous segment.
- Say: `delete that and write: Hello again` -> removes previous segment, then appends `Hello again`.
- Say: `don't include that` -> no transcript change.
- Say: `clear all` -> transcript becomes empty.

Environment variables
- `GROQ_API_KEY` (required): your Groq API key.
- `HOTKEY` (optional): hold-to-record key (default `f9`).
- `SMART_EDIT` (optional): `1` or `0` (default `1`).
- `SMART_MODE` (optional): `heuristic` or `llm` (default `llm`).
- `LLM_ALWAYS_CLEAN` (optional): in llm mode, clean every utterance (`1`) or only likely command edits (`0`) (default `1`).
- `PASTE_MODE` (optional): `delta` or `full` (default `delta`).
- `PASTE_BACKEND` (optional): `auto`, `x11`, `keyboard`, `pynput`, `xdotool`, `osascript` (default `auto`).
- `HOTKEY_BACKEND` (optional): `auto`, `x11`, `pynput`, `keyboard` (default `auto`).
- `RESET_TRANSCRIPT_EACH_RECORDING` (optional): `1` resets smart transcript on each hotkey press; `0` keeps session history (default `1`).
- `GROQ_WHISPER_MODEL` (optional): speech-to-text model (default `whisper-large-v3-turbo`).
- `GROQ_CHAT_MODEL` (optional): cleanup/intent model when `SMART_MODE=llm` (default `openai/gpt-oss-120b`).

Quality tuning (Whisper Flow style)
- Use:
```bash
SMART_MODE=llm
LLM_ALWAYS_CLEAN=1
GROQ_CHAT_MODEL=openai/gpt-oss-120b
GROQ_WHISPER_MODEL=whisper-large-v3-turbo
```
- `openai/gpt-oss-120b` is default for cleaner formatting and better self-corrections.
- `openai/gpt-oss-20b` is faster/cheaper but usually less consistent on cleanup quality.

Paste behavior
- `PASTE_MODE=delta` (default): only types the diff from previous transcript.
- If transcript shrinks (undo/clear), app sends backspaces for removed characters.
- If transcript grows, app pastes only the inserted characters.
- `PASTE_MODE=full`: select all (`Ctrl+A`) then paste full transcript.
- Linux fallback: app auto-tries native X11 injection first, then other backends.

Linux troubleshooting
- Run as your normal user first. Avoid `sudo` for desktop hotkey/paste workflows.
- If needed, install `xdotool` (`sudo apt install xdotool`) to improve paste/key injection fallback.
- If hotkey capture is unstable, force `HOTKEY_BACKEND=x11`.
- If paste is unstable, force `PASTE_BACKEND=x11` or `PASTE_BACKEND=xdotool`.

macOS troubleshooting
- Give Terminal accessibility permissions (System Settings -> Privacy & Security -> Accessibility and Input Monitoring).
- Auto mode prefers `pynput` for hotkeys and falls back to `keyboard`.
- Paste can fall back to AppleScript by setting `PASTE_BACKEND=osascript`.

About Groq API keys
- Visit https://www.groq.com (or the Groq developer console) to sign up and create an API key. Paste the key into your `.env` file as `GROQ_API_KEY`.
- Check the Groq documentation and pricing pages for current free tier or trial availability — policies and pricing can change.

Model and SDK
- This app uses the Groq SDK package (`groq`) and requests the `whisper-large-v3-turbo` model (see `transcribe_with_groq` in `app.py`).

Notes and troubleshooting
- If `sounddevice` fails to open your audio input: try selecting the correct input device or run with elevated permissions. See `sounddevice` docs.
- On some systems the GUI popup transparency may not be supported — this is optional and won't break the core functionality.
- If the `groq` package fails to import, ensure dependencies are installed and that your Python version is supported.

Contributing
- See `DEVELOPERS.md` for developer instructions, extension points, and ways to add tools/agents.

License
- This project is released under the MIT license (see `LICENSE`).

Dev Notes
- I made it pretty simple so anyone can build on top of it without restrictions - like giving it Agentic functions and allowing it to do local tasks. (Integration made easy and free)
- Since most vibe coders aren't willing to pay for tools like Whisper Flow and frankly, I wasn't going to pay for something I could get for free with the same precision.

Testing
- Install pytest as a dev dependency: `pip install pytest`
- Run tests: `pytest -q`

Questions / FAQ
- Q: Where is the transcription stored?
- A: Audio files are written to a temporary file and removed after processing. Transcription text is copied to clipboard and pasted into the active window.













# DictaPilot
# DictaPilot
