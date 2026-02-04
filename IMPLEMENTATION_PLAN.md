# DictaPilot PyQt6 GUI Implementation Plan

## 1. Objective
Convert the current CLI + popup workflow into a polished PyQt6 desktop app while preserving existing recording, smart dictation, and paste behavior.

## 2. Required Features
- Modern PyQt6 UI with a visually appealing layout.
- Settings for:
  - `GROQ_API_KEY`
  - transcription model (`GROQ_WHISPER_MODEL`)
  - cleanup model (`GROQ_CHAT_MODEL`)
  - smart mode (`heuristic` / `llm`)
- Left-side history panel with toggleable sidebar.
- Dark/Light theme toggle.

## 3. Current System (Reuse Plan)
- Keep these modules and logic:
  - `app.py` recording/transcription flow
  - `smart_editor.py` intent/cleanup pipeline
  - `paste_utils.py` paste backend logic
  - `x11_backend.py` Linux input backend
- Refactor app entry so core logic is UI-agnostic and callable from PyQt signals/slots.

## 4. Target Architecture
- `gui/main_window.py`: Main window, layout, toolbar, tab/panel controls.
- `gui/history_panel.py`: Left sidebar list, search/filter, toggle behavior.
- `gui/settings_dialog.py`: API key + model selectors + advanced toggles.
- `gui/theme.py`: light/dark palettes + QSS tokens.
- `gui/controller.py`: orchestrates recording, transcription, smart edit, paste, history updates.
- `core/session.py`: state model for transcript/history items.
- `main_qt.py`: PyQt6 bootstrap entrypoint.

## 5. UX Layout
- Top bar: app title, record status, theme toggle, settings button.
- Left: collapsible history sidebar (timestamp + snippet + action badge).
- Center: live transcript view + current output preview.
- Bottom: primary action row (Record/Hold Hotkey status, Clear, Copy, Paste mode indicator).
- Visual style:
  - rounded cards, subtle shadows, balanced spacing
  - high contrast text, readable typography
  - smooth transitions when sidebar expands/collapses

## 6. Settings Design
- Persist with `QSettings`:
  - API key (optionally keyring later)
  - model selections
  - theme mode
  - smart mode + cleanup flags
- Validate API key presence before transcription.
- Provide inline helper text for each model setting.

## 7. History Behavior
- Store each completed utterance as item:
  - timestamp
  - raw transcription
  - cleaned output
  - action (`append`, `undo`, `ignore`, etc.)
- Click history item to inspect details in center pane.
- Sidebar toggle:
  - expanded width: full details
  - collapsed width: icon rail

## 8. Migration Phases
1. Core Extraction
   - move reusable recording/transcription pipeline out of CLI-only flow.
2. Base Window
   - build main PyQt6 window with placeholder panels.
3. Functional Wiring
   - connect record flow, smart update, and paste.
4. Settings + Models
   - add settings dialog and persistent config.
5. History + Sidebar
   - implement list model and toggle behavior.
6. Theme System
   - add dark/light palettes and runtime switching.
7. QA + Packaging
   - Linux validation, test updates, docs.

## 9. Testing Strategy
- Unit tests:
  - settings serialization
  - history model behavior
  - theme switch state persistence
- Integration tests:
  - record -> transcribe -> smart edit -> UI update
  - sidebar toggle state
  - model selection reflected in runtime env/config
- Manual QA:
  - Linux desktop focus/paste behavior
  - long session history performance

## 10. Definition of Done
- PyQt6 app launches and supports all required settings.
- Sidebar history is toggleable and usable.
- Dark/light theme switch works instantly.
- Smart dictation flow remains intact with model selection support.
- README updated with PyQt6 run instructions.
