from smart_editor import TranscriptState, smart_update_state


def test_append_normal():
    state = TranscriptState()
    prev_out, new_out, action = smart_update_state(state, "Hello world.", "heuristic")

    assert prev_out == ""
    assert new_out == "Hello world."
    assert action == "append"


def test_undo_deletes_last_segment():
    state = TranscriptState()
    smart_update_state(state, "First sentence.", "heuristic")
    smart_update_state(state, "Second sentence.", "heuristic")

    prev_out, new_out, action = smart_update_state(state, "scratch that", "heuristic")

    assert prev_out == "First sentence. Second sentence."
    assert new_out == "First sentence."
    assert action == "undo"


def test_clear_all():
    state = TranscriptState()
    smart_update_state(state, "One", "heuristic")
    smart_update_state(state, "Two", "heuristic")

    prev_out, new_out, action = smart_update_state(state, "clear all", "heuristic")

    assert prev_out == "One. Two."
    assert new_out == ""
    assert action == "clear"


def test_ignore_utterance():
    state = TranscriptState()
    smart_update_state(state, "Keep this", "heuristic")

    prev_out, new_out, action = smart_update_state(state, "don't include that", "heuristic")

    assert prev_out == "Keep this."
    assert new_out == "Keep this."
    assert action == "ignore"


def test_ignore_it_variant():
    state = TranscriptState()
    smart_update_state(state, "Keep this too", "heuristic")

    prev_out, new_out, action = smart_update_state(state, "ignore it please", "heuristic")

    assert prev_out == "Keep this too."
    assert new_out == "Keep this too."
    assert action == "ignore"


def test_nevermind_variant():
    state = TranscriptState()
    smart_update_state(state, "Alpha beta", "heuristic")

    prev_out, new_out, action = smart_update_state(state, "never mind that", "heuristic")

    assert prev_out == "Alpha beta."
    assert new_out == "Alpha beta."
    assert action == "ignore"


def test_undo_plus_append_remainder():
    state = TranscriptState()
    smart_update_state(state, "Hello world.", "heuristic")
    smart_update_state(state, "Old ending.", "heuristic")

    prev_out, new_out, action = smart_update_state(
        state,
        "oh no delete that and write: Hello again.",
        "heuristic",
    )

    assert prev_out == "Hello world. Old ending."
    assert new_out == "Hello world. Hello again."
    assert action == "undo_append"


def test_clear_everything_variant():
    state = TranscriptState()
    smart_update_state(state, "One", "heuristic")
    smart_update_state(state, "Two", "heuristic")

    prev_out, new_out, action = smart_update_state(state, "clear everything", "heuristic")

    assert prev_out == "One. Two."
    assert new_out == ""
    assert action == "clear"


def test_inline_self_correction_replaces_prior_clause():
    state = TranscriptState()
    prev_out, new_out, action = smart_update_state(
        state,
        "Hello, how are you? My name is Rehan. No, no, my name is Numan, not Rehan.",
        "heuristic",
    )

    assert prev_out == ""
    assert new_out == "Hello, how are you? My name is Numan, not Rehan."
    assert action == "append"


def test_inline_no_sentence_without_overlap_is_kept():
    state = TranscriptState()
    prev_out, new_out, action = smart_update_state(
        state,
        "Let's leave now. No, I disagree.",
        "heuristic",
    )

    assert prev_out == ""
    assert new_out == "Let's leave now. No, I disagree."
    assert action == "append"


def test_inline_not_use_rewrites_single_clause():
    state = TranscriptState()
    prev_out, new_out, action = smart_update_state(
        state,
        "Now your task is to build a solid plan with tasks to change it into GUI using Twinker, not Twinker, PyQT6.",
        "heuristic",
    )

    assert prev_out == ""
    assert new_out == "Now your task is to build a solid plan with tasks to change it into GUI using PyQT6."
    assert action == "append"


def test_inline_not_use_rewrites_previous_clause():
    state = TranscriptState()
    prev_out, new_out, action = smart_update_state(
        state,
        "Build a solid plan for convert this into GUI using Twinkler. Oh no no not Twinkler use PYQT",
        "heuristic",
    )

    assert prev_out == ""
    assert new_out == "Build a solid plan for convert this into GUI using PYQT."
    assert action == "append"


def test_filler_and_repetition_cleanup():
    state = TranscriptState()
    prev_out, new_out, action = smart_update_state(state, "um um this this is is a test", "heuristic")

    assert prev_out == ""
    assert new_out == "This is a test."
    assert action == "append"


def test_question_mark_auto_punctuation():
    state = TranscriptState()
    prev_out, new_out, action = smart_update_state(state, "how are you", "heuristic")

    assert prev_out == ""
    assert new_out == "How are you?"
    assert action == "append"
