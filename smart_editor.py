"""
DictaPilot Smart Editor
Handles smart dictation commands like delete, clear, ignore, replace.

MIT License
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

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


COMMAND_KEYWORDS = (
    "delete",
    "undo",
    "scratch",
    "remove",
    "erase",
    "drop that",
    "take that out",
    "clear",
    "reset",
    "start over",
    "clear everything",
    "don't include",
    "do not include",
    "don't add",
    "do not add",
    "ignore",
    "ignore it",
    "skip",
    "disregard",
    "omit",
    "leave that out",
    "cancel that",
    "nevermind",
    "never mind",
)
VALID_ACTIONS = {"append", "undo", "undo_append", "clear", "ignore"}
QUESTION_STARTERS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "which",
    "is",
    "are",
    "am",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "have",
    "has",
    "had",
}
EXCLAMATION_STARTERS = {"wow", "great", "awesome", "amazing", "congrats", "congratulations"}

_COMMAND_PREFACE_RE = re.compile(
    r"^(?:(?:oh no|oops|please|hey|ok(?:ay)?|wait|well|uh|um|hmm)\b[\s,\-.:;]*)+",
    re.IGNORECASE,
)
_CONTENT_FILLER_RE = re.compile(r"^(?:(?:uh|um|erm|ah|hmm)\b[\s,\-.:;]*)+", re.IGNORECASE)
_UNDO_RE = re.compile(
    r"^(?:delete that|delete previous|undo(?: that)?|scratch that|remove that|remove previous|"
    r"take that out|erase that|drop that|backspace that)\b(?P<rest>.*)$",
    re.IGNORECASE,
)
_CLEAR_RE = re.compile(
    r"^(?:clear all|clear everything|reset(?: all)?|start over|wipe all|wipe everything|erase all)\b[\s,.!?:;-]*$",
    re.IGNORECASE,
)
_CLEAR_SIMPLE_RE = re.compile(
    r"^(?:clear|reset|wipe)\b[\s,.!?:;-]*$",
    re.IGNORECASE,
)
_IGNORE_RE = re.compile(
    r"^(?:"
    r"don['']t include(?: that| this| it)?|"
    r"do not include(?: that| this| it)?|"
    r"don['']t add(?: that| this| it)?|"
    r"do not add(?: that| this| it)?|"
    r"ignore(?: that| this| it)?|"
    r"skip(?: that| this| it)?|"
    r"disregard(?: that| this| it)?|"
    r"omit(?: that| this| it)?|"
    r"leave (?:that|this|it) out|"
    r"cancel that|"
    r"never ?mind(?: that| this| it)?"
    r")\b.*$",
    re.IGNORECASE,
)
_IGNORE_TRAILING_RE = re.compile(
    r"^(?P<before>.*?)\b(?:"
    r"ignore(?: that| this| it)?|"
    r"skip(?: that| this| it)?|"
    r"disregard(?: that| this| it)?|"
    r"omit(?: that| this| it)?|"
    r"don't include(?: that| this| it)?|"
    r"do not include(?: that| this| it)?|"
    r"don't add(?: that| this| it)?|"
    r"do not add(?: that| this| it)?|"
    r"never ?mind(?: that| this| it)?"
    r")\s*$",
    re.IGNORECASE,
)
_INLINE_CORRECTION_RE = re.compile(
    r"^\s*(?:(?:oh|uh|um)\s+)*(?:no(?:\s*,?\s*no)*|nope|sorry|i mean|actually)\b[\s,:\-]*",
    re.IGNORECASE,
)
_USE_NOT_USE_INLINE_RE = re.compile(
    r"\b(?P<lemma>use|using)\s+"
    r"(?P<wrong>[A-Za-z0-9][A-Za-z0-9+#.\-]*)\s*,?\s*"
    r"not\s+(?P=wrong)\s*,?\s*"
    r"(?:(?:use|using|with)\s+)?"
    r"(?P<right>[A-Za-z0-9][A-Za-z0-9+#.\-]*)",
    re.IGNORECASE,
)
_NEGATION_REPLACEMENT_RE = re.compile(
    r"^not\s+(?P<wrong>[A-Za-z0-9][A-Za-z0-9+#.\-]*)\s*"
    r"(?:,|\s)*(?:(?:use|using|with|but|instead|rather)\s+)?"
    r"(?P<right>[A-Za-z0-9][A-Za-z0-9+#.\-]*)[.?!]?$",
    re.IGNORECASE,
)
_FILLER_WORD_RE = re.compile(r"\b(?:uh+|um+|erm+|ah+|hmm+|mm+)\b", re.IGNORECASE)
_FILLER_PHRASE_RE = re.compile(r"\b(?:you know|i mean|kind of|sort of)\b", re.IGNORECASE)
_REPEATED_WORD_RE = re.compile(r"\b(?P<word>[A-Za-z][A-Za-z0-9']*)\b(?:\s+(?P=word)\b)+", re.IGNORECASE)
_REPEATED_PUNCT_RE = re.compile(r"([,.;:!?])\1+")
_REPLACE_RE = re.compile(
    r"^(?:replace|change|swap)\s+(?P<target>.+?)\s+(?:with|to|for)\s+(?P<replacement>.+)$",
    re.IGNORECASE,
)


@dataclass
class TranscriptState:
    segments: List[str] = field(default_factory=list)
    output_text: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _strip_command_preface(text: str) -> str:
    return _COMMAND_PREFACE_RE.sub("", _normalize_spaces(text), count=1).strip()


def _normalize_segment(text: str) -> str:
    cleaned = _normalize_spaces(text)
    cleaned = _CONTENT_FILLER_RE.sub("", cleaned, count=1).strip()
    cleaned = _cleanup_disfluencies(cleaned)
    return _polish_punctuation(cleaned)


def _join_segments(segments: List[str]) -> str:
    return " ".join(part.strip() for part in segments if part and part.strip()).strip()


def _replace_last_case_insensitive(text: str, target: str, replacement: str) -> str:
    if not text or not target:
        return text
    pattern = re.compile(rf"\b{re.escape(target)}\b", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    last = matches[-1]
    return f"{text[: last.start()]}{replacement}{text[last.end() :]}"


def _rewrite_not_use_inline(text: str) -> str:
    def _replacement(match: re.Match) -> str:
        lemma = match.group("lemma")
        right = match.group("right")
        return f"{lemma} {right}"

    return _USE_NOT_USE_INLINE_RE.sub(_replacement, text)


def _clean_remainder(text: str) -> str:
    rest = _normalize_spaces(text)
    rest = rest.lstrip(" ,.:;!-")
    rest = re.sub(r"^(?:and|then|instead)\b[\s,:-]*", "", rest, flags=re.IGNORECASE)
    rest = re.sub(
        r"^(?:and\s+)?(?:please\s+)?(?:write|say|type)\b[\s,:-]*",
        "",
        rest,
        flags=re.IGNORECASE,
    )
    rest = re.sub(r"^(?:and|then)\b[\s,:-]*", "", rest, flags=re.IGNORECASE)
    return _normalize_segment(rest)


def _cleanup_disfluencies(text: str) -> str:
    cleaned = _normalize_spaces(text)
    if not cleaned:
        return ""
    cleaned = _FILLER_PHRASE_RE.sub(" ", cleaned)
    cleaned = _FILLER_WORD_RE.sub(" ", cleaned)
    cleaned = _REPEATED_WORD_RE.sub(lambda m: m.group("word"), cleaned)
    cleaned = _REPEATED_PUNCT_RE.sub(lambda m: m.group(1), cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _terminal_punctuation(text: str) -> str:
    if not text:
        return "."
    first_match = re.search(r"[A-Za-z']+", text)
    first_word = first_match.group(0).lower() if first_match else ""
    if first_word in QUESTION_STARTERS:
        return "?"
    if first_word in EXCLAMATION_STARTERS:
        return "!"
    return "."


def _capitalize_sentences(text: str) -> str:
    if not text:
        return ""
    chars = list(text)
    capitalize_next = True
    for idx, ch in enumerate(chars):
        if capitalize_next and ch.isalpha():
            chars[idx] = ch.upper()
            capitalize_next = False
        if ch in ".!?":
            capitalize_next = True
    return "".join(chars)


def _polish_punctuation(text: str) -> str:
    cleaned = _normalize_spaces(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", cleaned)
    cleaned = _capitalize_sentences(cleaned)
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += _terminal_punctuation(cleaned)
    return cleaned


def _significant_tokens(text: str) -> set:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "have",
        "from",
        "your",
        "just",
        "really",
        "very",
        "like",
        "then",
        "there",
        "here",
        "what",
        "when",
        "where",
        "which",
        "would",
        "could",
        "should",
        "into",
        "about",
        "been",
        "were",
        "they",
        "them",
        "their",
        "i",
        "you",
        "we",
        "he",
        "she",
        "it",
        "my",
        "our",
        "his",
        "her",
        "its",
        "not",
    }
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in stopwords}


def _rewrite_previous_clause(previous: str, correction: str) -> Optional[str]:
    match = _NEGATION_REPLACEMENT_RE.match(_normalize_spaces(correction))
    if not match:
        return None

    wrong = match.group("wrong")
    right = match.group("right")
    rewritten = _replace_last_case_insensitive(previous, wrong, right)
    if rewritten == previous:
        return None
    return _normalize_spaces(rewritten)


def _apply_inline_corrections(text: str) -> str:
    text = _rewrite_not_use_inline(_normalize_spaces(text))
    clauses = [part.strip() for part in re.findall(r"[^.?!]+[.?!]?", text) if part.strip()]
    if not clauses:
        return text

    corrected = []
    for clause in clauses:
        cleaned_clause = _normalize_spaces(clause)
        marker_match = _INLINE_CORRECTION_RE.match(cleaned_clause)
        if not marker_match or not corrected:
            corrected.append(cleaned_clause)
            continue

        remainder = _normalize_spaces(cleaned_clause[marker_match.end() :])
        remainder = _rewrite_not_use_inline(remainder)
        if not remainder:
            continue

        rewritten_previous = _rewrite_previous_clause(corrected[-1], remainder)
        if rewritten_previous:
            corrected[-1] = rewritten_previous
            continue

        previous_tokens = _significant_tokens(corrected[-1])
        remainder_tokens = _significant_tokens(remainder)
        if previous_tokens & remainder_tokens:
            # Keep sentence starts readable when replacement begins with lowercase.
            for idx, ch in enumerate(remainder):
                if ch.isalpha():
                    if ch.islower():
                        remainder = remainder[:idx] + ch.upper() + remainder[idx + 1 :]
                    break
            corrected[-1] = remainder
        else:
            corrected.append(cleaned_clause)

    return _normalize_spaces(_rewrite_not_use_inline(" ".join(corrected)))


def needs_intent_handling(utterance: str) -> bool:
    lowered = _normalize_spaces(utterance).lower()
    
    if not lowered:
        return False
    
    command_text = _strip_command_preface(lowered)
    
    ignore_keywords = {"ignore", "skip", "disregard", "omit", "don't include", "do not include", 
                       "don't add", "do not add", "never mind", "cancel"}
    
    if any(kw in command_text for kw in ignore_keywords):
        return True
    
    clear_keywords = {"clear", "reset", "wipe", "erase", "start over"}
    if any(kw in command_text for kw in clear_keywords):
        return True
    
    undo_keywords = {"delete", "undo", "scratch", "remove", "erase", "drop", "take that out"}
    if any(kw in command_text for kw in undo_keywords):
        return True
    
    return False


def _handle_replace_pattern(state: TranscriptState, target: str, replacement: str) -> Tuple[str, str]:
    target_cleaned = _normalize_spaces(target)
    replacement_cleaned = _normalize_segment(replacement)
    
    if not target_cleaned or not state.output_text:
        return state.output_text, "ignore"
    
    new_output = _replace_last_case_insensitive(state.output_text, target_cleaned, replacement_cleaned)
    
    if new_output != state.output_text:
        state.output_text = new_output
        state.segments = [new_output] if new_output else []
        return state.output_text, "undo_append"
    
    return state.output_text, "ignore"


def _handle_delete_specific(state: TranscriptState, target: str) -> Tuple[str, str]:
    target_cleaned = _normalize_spaces(target)
    
    if not target_cleaned or not state.output_text:
        return state.output_text, "ignore"
    
    pattern = re.compile(rf"\b{re.escape(target_cleaned)}\b", re.IGNORECASE)
    new_output = pattern.sub("", state.output_text)
    new_output = re.sub(r"\s+", " ", new_output).strip()
    
    if new_output != state.output_text:
        state.output_text = new_output
        state.segments = [new_output] if new_output else []
        return state.output_text, "undo"
    
    return state.output_text, "ignore"


def apply_heuristic(state: TranscriptState, utterance: str) -> Tuple[str, str]:
    raw = _normalize_spaces(utterance)
    if not raw:
        return state.output_text, "ignore"

    command_text = _strip_command_preface(raw)

    if _CLEAR_RE.match(command_text) or _CLEAR_SIMPLE_RE.match(command_text):
        state.segments.clear()
        state.output_text = ""
        return state.output_text, "clear"

    replace_match = _REPLACE_RE.match(command_text)
    if replace_match:
        target = _normalize_spaces(replace_match.group("target") or "")
        replacement = _normalize_spaces(replace_match.group("replacement") or "")
        return _handle_replace_pattern(state, target, replacement)

    trailing_ignore_match = _IGNORE_TRAILING_RE.match(command_text)
    if trailing_ignore_match:
        before = _normalize_spaces(trailing_ignore_match.group("before") or "")
        if before:
            return state.output_text, "ignore"
        return state.output_text, "ignore"

    if _IGNORE_RE.match(command_text):
        return state.output_text, "ignore"

    undo_match = _UNDO_RE.match(command_text)
    if undo_match:
        if state.segments:
            state.segments.pop()
        remainder = _clean_remainder(undo_match.group("rest") or "")
        action = "undo"
        if remainder:
            state.segments.append(remainder)
            action = "undo_append"
        state.output_text = _join_segments(state.segments)
        return state.output_text, action

    segment = _normalize_segment(_apply_inline_corrections(raw))
    if not segment:
        return state.output_text, "ignore"

    state.segments.append(segment)
    state.output_text = _join_segments(state.segments)
    return state.output_text, "append"


def _extract_json_object(raw: str) -> Optional[dict]:
    if not raw:
        return None

    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    end = -1
    for idx in range(start, len(raw)):
        char = raw[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end < 0:
        return None

    snippet = raw[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalize_llm_result(prev_output: str, updated: str, action: str) -> Tuple[str, str]:
    normalized_action = _normalize_spaces(action or "").lower()
    
    if normalized_action not in VALID_ACTIONS:
        if prev_output != updated:
            normalized_action = "append"
        else:
            normalized_action = "ignore"
    
    if normalized_action == "clear":
        return "", "clear"
    
    if normalized_action == "ignore":
        return prev_output, "ignore"
    
    if normalized_action in {"undo", "undo_append"}:
        if not updated:
            return "", normalized_action
        return updated, normalized_action
    
    normalized_output = _polish_punctuation(_cleanup_disfluencies(updated or ""))
    
    if normalized_output != prev_output:
        return normalized_output, "append"
    
    return prev_output, "ignore"


def _llm_updated_transcript(prev_output: str, utterance: str) -> Optional[Tuple[str, str]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq
    except Exception:
        return None

    model = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=api_key)

    system_prompt = (
        "You are DictaPilot Smart Editor - a precise voice dictation assistant. "
        "Your goal is to convert raw speech into clean, polished text while respecting user commands. "
        "Return JSON only with keys: updated_transcript, action. "
        "Valid actions: append, undo, undo_append, clear, ignore. "
        "Core rules:"
        "- Remove filler words (uh, um, you know, i mean) and repeated words"
        "- Apply proper punctuation (capitalization, periods, commas)"
        "- Preserve names, technical terms, and important content"
        "- Handle self-corrections intelligently (e.g., 'no not X use Y')"
        "- For undo commands, remove the last added content"
        "- For clear commands, return empty transcript with action 'clear'"
        "- For ignore commands, keep transcript unchanged"
        "- For 'X ignore' or 'ignore X' patterns, discard X and keep transcript unchanged"
    )
    user_prompt = (
        f"Current transcript:\n---\n{prev_output}\n---\n\n"
        f"New utterance:\n---\n{utterance}\n---\n\n"
        "Apply smart dictation behavior:\n"
        "1. COMMANDS - Handle these first:\n"
        "   - undo/delete/scratch/remove previous: Remove last segment from transcript\n"
        "   - clear/reset/start over: Return EMPTY transcript with action 'clear'\n"
        "   - ignore/skip/disregard/don't include: Keep transcript UNCHANGED\n"
        "   - 'X ignore' or 'ignore X' patterns: Discard X, keep transcript unchanged\n"
        "   - 'delete X' where X is specific text: Remove X from transcript\n"
        "   - 'replace X with Y': Replace X with Y in transcript\n\n"
        "2. CONTENT - If not a command:\n"
        "   - Clean up disfluencies (uh, um, repeated words)\n"
        "   - Add proper punctuation\n"
        "   - Append cleaned text to transcript\n\n"
        "3. OUTPUT:\n"
        "   - updated_transcript: FULL final transcript after applying this utterance\n"
        "   - action: One of append, undo, undo_append, clear, ignore\n"
        "   - If ignoring, updated_transcript should equal the previous transcript\n"
        "   - If clearing, updated_transcript should be empty string\n"
        "   - For undo/undo_append, include the remaining transcript after removal\n"
    )

    try:
        request = dict(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        try:
            resp = client.chat.completions.create(**request)
        except TypeError:
            request.pop("response_format", None)
            resp = client.chat.completions.create(**request)
        content = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None

    data = _extract_json_object(content)
    if not data:
        return None

    updated = data.get("updated_transcript")
    action = str(data.get("action") or "").strip().lower()

    if not isinstance(updated, str):
        return None

    return _normalize_llm_result(prev_output, updated, action)


def _sync_segments_from_output(state: TranscriptState, prev_output: str, new_output: str) -> None:
    if not new_output:
        state.segments = []
        return

    if _join_segments(state.segments) != prev_output:
        state.segments = [prev_output] if prev_output else []

    if new_output.startswith(prev_output):
        inserted = _normalize_spaces(new_output[len(prev_output) :]).strip()
        if inserted:
            state.segments.append(inserted)
        elif not state.segments:
            state.segments = [new_output]
        return

    if prev_output.startswith(new_output):
        while state.segments and _join_segments(state.segments) != new_output:
            state.segments.pop()
        if _join_segments(state.segments) != new_output:
            state.segments = [new_output] if new_output else []
        return

    state.segments = [new_output]


def smart_update_state(state: TranscriptState, utterance: str, mode: str = "heuristic") -> Tuple[str, str, str]:
    selected_mode = (mode or "heuristic").strip().lower()
    if selected_mode not in {"heuristic", "llm"}:
        selected_mode = "heuristic"

    with state.lock:
        prev_output = state.output_text

        llm_always_clean = _env_flag("LLM_ALWAYS_CLEAN", "1")
        use_llm = selected_mode == "llm" and (llm_always_clean or needs_intent_handling(utterance))
        if use_llm:
            llm_result = _llm_updated_transcript(prev_output, utterance)
            if llm_result is not None:
                new_output, action = llm_result
                if action == "ignore":
                    return prev_output, prev_output, "ignore"
                _sync_segments_from_output(state, prev_output, new_output)
                state.output_text = new_output
                return prev_output, state.output_text, action

        new_output, action = apply_heuristic(state, utterance)
        return prev_output, new_output, action
