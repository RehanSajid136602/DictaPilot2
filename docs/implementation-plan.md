# DictaPilot Feature Implementation Plan

## Goal
Implement the two highest-priority features:
1. Selected-Text Command Mode
2. Snippet + Dictionary Manager UI

## Scope
- Keep current hold-to-dictate flow unchanged.
- Reuse existing architecture (`app.py`, `smart_editor.py`, `paste_utils.py`, `tray.py`, `config.py`).
- Add tests for new behavior and remove CI false-positive test pass.

## Feature 1: Selected-Text Command Mode

### User Outcome
User selects text, triggers command hotkey, speaks a command (rewrite/fix/translate/shorten), and selected text is replaced in place.

### Tasks
- [ ] Add config/env flags:
  - [ ] `COMMAND_MODE_ENABLED=1`
  - [ ] `COMMAND_HOTKEY=f10`
  - [ ] `COMMAND_TIMEOUT_MS=8000`
  - [ ] Update `.env.example` and `README.md`.
- [ ] Register second hotkey flow in `app.py` using existing `HotkeyManager`.
- [ ] Build selection capture utility:
  - [ ] Save current clipboard.
  - [ ] Send copy shortcut to active app.
  - [ ] Read selected text from clipboard.
  - [ ] Restore original clipboard.
- [ ] Add command-mode speech capture lifecycle in `app.py`:
  - [ ] UI states: `Capturing Selection`, `Listening`, `Transforming`, `Applied`, `Error`.
  - [ ] Timeout and empty-selection handling.
- [ ] Add transform function in `smart_editor.py`:
  - [ ] `llm_transform_selected(selected_text, voice_command, context)`.
  - [ ] Reuse Groq JSON response style.
  - [ ] Enforce safe output contract (string result, fallback on invalid response).
- [ ] Replace selected text via existing paste backend path (`paste_utils.py`).
- [ ] Add logging + storage entry for command-mode action (new action type if needed).

### Tests
- [ ] Add parser/intention tests for command utterances.
- [ ] Add transform result normalization tests (valid/invalid JSON).
- [ ] Add mocked integration test for selection -> transform -> replace pipeline.

### Done Criteria
- [ ] Works on Linux/macOS/Windows with backend fallback.
- [ ] No regression in standard dictation flow.
- [ ] Command mode can be disabled via env/config.

## Feature 2: Snippet + Dictionary Manager UI

### User Outcome
User can manage snippets and personal dictionary from UI (add/edit/delete/import/export) without editing JSON files manually.

### Tasks
- [ ] Add settings window module (Tkinter) with tabs:
  - [ ] `Dictionary`
  - [ ] `Snippets`
- [ ] Build CRUD operations:
  - [ ] Add new item
  - [ ] Edit item
  - [ ] Delete item
  - [ ] Search/filter list
- [ ] Build import/export actions:
  - [ ] Import JSON (validate schema)
  - [ ] Export JSON
- [ ] Integrate with existing path conventions from `smart_editor.py`:
  - [ ] `dictionary.json`
  - [ ] `snippets.json`
- [ ] Use atomic save (temp file + replace) to prevent data corruption.
- [ ] Surface clear validation errors (empty key, duplicate key, invalid JSON).
- [ ] Wire settings entry from tray menu (`tray.py`) to open this manager.
- [ ] Keep existing config fields (hotkey/model/backend) available in same settings flow.

### Tests
- [ ] Add loader/saver tests for dictionary/snippets helpers.
- [ ] Add duplicate-key and malformed JSON tests.
- [ ] Add schema compatibility tests for dict and list JSON formats.

### Done Criteria
- [ ] User can fully manage dictionary/snippets from UI.
- [ ] UI edits are reflected immediately in smart editor behavior.
- [ ] Existing manual JSON format remains backward compatible.

## Cross-Cutting Work
- [ ] CI hardening:
  - [ ] Remove `pytest -q || true` from `.github/workflows/ci.yml`.
  - [ ] Ensure tests fail CI on real failures.
- [ ] Documentation:
  - [ ] Add command-mode usage examples.
  - [ ] Add settings manager usage guide with screenshots/GIF.
  - [ ] Add troubleshooting for clipboard/input permissions.
- [ ] Telemetry/logging (local only):
  - [ ] Command mode success/failure counters.
  - [ ] Transform latency timing.

## Suggested Delivery Order
1. Command mode backend + tests
2. Command mode UX polish + docs
3. Manager UI + tray wiring
4. Manager validation/import-export + tests
5. CI strictness + release notes

## Risks and Mitigations
- Clipboard race conditions:
  - Mitigation: save/restore clipboard and add bounded retry.
- Platform input permission failures:
  - Mitigation: backend fallback and explicit error guidance.
- LLM output drift:
  - Mitigation: strict JSON contract + safe fallback behavior.

