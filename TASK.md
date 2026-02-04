# DictaPilot PyQt6 Task List

## Phase 0: Project Setup
- [ ] Add PyQt6 dependency and verify Linux install.
- [ ] Create `gui/` package and `main_qt.py` entrypoint.
- [ ] Add feature flag to choose CLI or PyQt startup mode.

## Phase 1: Core Refactor
- [ ] Extract recording + transcription flow into reusable controller/service.
- [ ] Keep `smart_editor.py` and `paste_utils.py` integration unchanged.
- [ ] Replace direct print/log calls with signal-friendly status events.

## Phase 2: Main Window
- [ ] Build `QMainWindow` with:
  - [ ] top header/status bar
  - [ ] left history sidebar
  - [ ] center transcript/output view
  - [ ] bottom actions row
- [ ] Add polished spacing, card styling, and consistent typography.

## Phase 3: History Sidebar
- [ ] Implement history item model (timestamp, action, preview text).
- [ ] Add sidebar list with click-to-preview details.
- [ ] Add toggle button for collapse/expand behavior.
- [ ] Persist sidebar expanded/collapsed state.

## Phase 4: Settings UI
- [ ] Create settings dialog/page for:
  - [ ] `GROQ_API_KEY`
  - [ ] `GROQ_WHISPER_MODEL`
  - [ ] `GROQ_CHAT_MODEL`
  - [ ] `SMART_MODE`
  - [ ] `LLM_ALWAYS_CLEAN`
- [ ] Persist settings via `QSettings`.
- [ ] Add validation + user feedback for missing/invalid values.

## Phase 5: Theme System
- [ ] Implement dark and light themes (palette + QSS tokens).
- [ ] Add theme toggle in header.
- [ ] Persist and restore user theme preference.

## Phase 6: Flow Integration
- [ ] Connect record action/hotkey state to UI indicators.
- [ ] Show raw transcription, smart action, and cleaned transcript in UI.
- [ ] Ensure paste still targets focused external input as before.
- [ ] Ensure undo/clear/ignore commands update UI and history correctly.

## Phase 7: Quality & Testing
- [ ] Add unit tests for settings persistence and history model.
- [ ] Add integration tests for controller + UI update sequence.
- [ ] Run full regression tests for existing smart editor behavior.
- [ ] Manual Linux QA: permissions, hotkey backend, paste backend.

## Phase 8: Docs & Release
- [ ] Update `README.md` with PyQt6 setup and run commands.
- [ ] Document model selection recommendations.
- [ ] Add migration notes from CLI flow to GUI flow.
- [ ] Prepare release checklist and tagged version.

## Acceptance Checklist
- [ ] Visually appealing PyQt6 UI is complete.
- [ ] Model/API settings are editable in-app.
- [ ] Left history tab exists and is toggleable.
- [ ] Dark/light theme toggle works and persists.
- [ ] Current smart dictation quality/features are preserved.
