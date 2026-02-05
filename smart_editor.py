"""
DictaPilot Smart Editor
Handles smart dictation commands like delete, clear, ignore, replace.

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

import json
import os
import re
import threading
import platform
from pathlib import Path

from app_context import DictationContext, get_context, update_profile
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
DICTATION_MODE = (os.getenv("DICTATION_MODE") or "accurate").strip().lower()
CLEANUP_LEVEL = (os.getenv("CLEANUP_LEVEL") or "aggressive").strip().lower()
if DICTATION_MODE not in {"speed", "balanced", "accurate"}:
    DICTATION_MODE = "balanced"
if CLEANUP_LEVEL not in {"basic", "balanced", "aggressive"}:
    CLEANUP_LEVEL = "balanced"
PERSONAL_DICTIONARY_PATH = (os.getenv("PERSONAL_DICTIONARY_PATH") or "").strip()
SNIPPETS_PATH = (os.getenv("SNIPPETS_PATH") or "").strip()
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
_AGGRESSIVE_FILLER_RE = re.compile(
    r"\b(?:basically|literally|honestly|actually|like)\b",
    re.IGNORECASE,
)
_REPEATED_WORD_RE = re.compile(r"\b(?P<word>[A-Za-z][A-Za-z0-9']*)\b(?:\s+(?P=word)\b)+", re.IGNORECASE)
_REPEATED_PUNCT_RE = re.compile(r"([,.;:!?])\1+")
_REPLACE_RE = re.compile(
    r"^(?:replace|change|swap)\s+(?P<target>.+?)\s+(?:with|to|for)\s+(?P<replacement>.+)$",
    re.IGNORECASE,
)
_REWRITE_RE = re.compile(
    r"^(?:rewrite|rephrase|polish|improve|make(?: it)?)\s+(?P<tone>formal|polite|casual|friendly|concise|professional|natural)?\b.*$",
    re.IGNORECASE,
)
_GRAMMAR_RE = re.compile(r"^(?:fix|correct)\s+(?:the\s+)?(grammar|spelling|punctuation)\b", re.IGNORECASE)
_TONE_SET_RE = re.compile(r"^(?:tone|style|voice)\s+(?P<tone>.+)$", re.IGNORECASE)
_LANG_SET_RE = re.compile(r"^(?:language|lang)\s+(?P<lang>.+)$", re.IGNORECASE)
_POLITE_REWRITE_RE = re.compile(r"\b(make it|rewrite|rephrase)\s+more\s+(polite|formal|casual|friendly)\b", re.IGNORECASE)
_FORMAL_REWRITE_RE = re.compile(r"\b(make it|rewrite|rephrase)\s+formal\b", re.IGNORECASE)
_POLITE_ONLY_RE = re.compile(r"^(?:polite|formal|casual|friendly|concise)\s+tone\b", re.IGNORECASE)


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
    if CLEANUP_LEVEL == "aggressive":
        cleaned = _AGGRESSIVE_FILLER_RE.sub(" ", cleaned)
    cleaned = _REPEATED_WORD_RE.sub(lambda m: m.group("word"), cleaned)
    cleaned = _REPEATED_PUNCT_RE.sub(lambda m: m.group(1), cleaned)
    if CLEANUP_LEVEL == "aggressive":
        cleaned = _dedupe_repeated_phrases(cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _dedupe_repeated_phrases(text: str) -> str:
    pattern = re.compile(r"\b(?P<phrase>\w+(?:\s+\w+){1,3})\b(?:\s+(?P=phrase)\b)+", re.IGNORECASE)
    return pattern.sub(lambda m: m.group("phrase"), text)


def _default_data_path(filename: str) -> Optional[Path]:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("APPDATA")
        if not base:
            return None
        return Path(base) / "DictaPilot" / filename
    return Path.home() / ".config" / "dictapilot" / filename


_DICTIONARY_CACHE = {"path": None, "mtime": None, "data": {}}
_SNIPPETS_CACHE = {"path": None, "mtime": None, "data": {}}
_DEFAULT_DICTIONARY = {"adelant": "Adelant"}


def _load_json_map(path: Optional[Path], cache: dict) -> dict:
    if path is None:
        return {}
    path_str = str(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        cache.update({"path": path_str, "mtime": None, "data": {}})
        return {}

    if cache.get("path") == path_str and cache.get("mtime") == mtime:
        return cache.get("data", {})

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}

    data = {}
    if isinstance(raw, dict):
        data = {str(k).strip().lower(): str(v) for k, v in raw.items() if str(k).strip()}
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                key = str(item.get("key") or item.get("from") or "").strip().lower()
                val = str(item.get("value") or item.get("to") or "").strip()
                if key:
                    data[key] = val

    cache.update({"path": path_str, "mtime": mtime, "data": data})
    return data


def _personal_dictionary() -> dict:
    path = Path(PERSONAL_DICTIONARY_PATH) if PERSONAL_DICTIONARY_PATH else _default_data_path("dictionary.json")
    data = _load_json_map(path, _DICTIONARY_CACHE)
    merged = dict(_DEFAULT_DICTIONARY)
    merged.update(data)
    return merged


def _snippets() -> dict:
    path = Path(SNIPPETS_PATH) if SNIPPETS_PATH else _default_data_path("snippets.json")
    return _load_json_map(path, _SNIPPETS_CACHE)


def _apply_personal_dictionary(text: str) -> str:
    mapping = _personal_dictionary()
    if not mapping or not text:
        return text

    updated = text
    for key, value in mapping.items():
        if not key:
            continue
        if re.match(r"^[A-Za-z0-9]+$", key):
            pattern = re.compile(rf"\\b{re.escape(key)}\\b", re.IGNORECASE)
            updated = pattern.sub(value, updated)
        else:
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            updated = pattern.sub(value, updated)
    return updated


_SNIPPET_RE = re.compile(r"^(?:insert|snippet|shortcut)\\s+(?P<name>.+)$", re.IGNORECASE)


def _resolve_snippet(utterance: str) -> Optional[str]:
    match = _SNIPPET_RE.match(_normalize_spaces(utterance))
    if not match:
        return None
    name = (match.group("name") or "").strip().lower()
    if not name:
        return None
    mapping = _snippets()
    return mapping.get(name)


def _detect_transform_command(utterance: str) -> Optional[dict]:
    normalized = _normalize_spaces(utterance)
    if not normalized:
        return None
    grammar = _GRAMMAR_RE.match(normalized)
    if grammar:
        return {"type": "grammar"}
    rewrite = _REWRITE_RE.match(normalized)
    if rewrite:
        tone = (rewrite.group("tone") or "").strip().lower()
        if not tone:
            polite = _POLITE_REWRITE_RE.search(normalized)
            if polite:
                tone = polite.group(2).lower()
        if not tone and _FORMAL_REWRITE_RE.search(normalized):
            tone = "formal"
        return {"type": "rewrite", "tone": tone}
    return None


def is_transform_command(utterance: str) -> bool:
    return _detect_transform_command(utterance) is not None


def _detect_setting_command(utterance: str) -> Optional[dict]:
    normalized = _normalize_spaces(utterance)
    if not normalized:
        return None
    tone = _TONE_SET_RE.match(normalized)
    if tone:
        return {"type": "tone", "value": (tone.group("tone") or "").strip().lower()}
    if _POLITE_ONLY_RE.match(normalized):
        return {"type": "tone", "value": normalized.split()[0].strip().lower()}
    lang = _LANG_SET_RE.match(normalized)
    if lang:
        return {"type": "language", "value": (lang.group("lang") or "").strip().lower()}
    return None


def llm_refine(prev_output: str, utterance: str, context: Optional[DictationContext] = None) -> Optional[Tuple[str, str]]:
    ctx = context or get_context()
    intent = _detect_transform_command(utterance)
    return _llm_updated_transcript(prev_output, utterance, ctx.tone, ctx.language, intent)


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


def _dedupe_overlap(prev_output: str, segment: str) -> str:
    if not prev_output or not segment:
        return segment
    prev_tokens = prev_output.split()
    seg_tokens = segment.split()
    if not prev_tokens or not seg_tokens:
        return segment
    max_check = min(3, len(prev_tokens), len(seg_tokens))
    for size in range(max_check, 0, -1):
        if prev_tokens[-size:] == seg_tokens[:size]:
            return " ".join(seg_tokens[size:]).strip()
    return segment


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
    raw = _normalize_spaces(_apply_personal_dictionary(utterance))
    if not raw:
        return state.output_text, "ignore"

    setting_cmd = _detect_setting_command(raw)
    if setting_cmd:
        ctx = get_context()
        if setting_cmd["type"] == "tone":
            update_profile(ctx.app_id, tone=setting_cmd["value"])
        if setting_cmd["type"] == "language":
            update_profile(ctx.app_id, language=setting_cmd["value"])
        return state.output_text, "ignore"

    snippet = _resolve_snippet(raw)
    if snippet:
        snippet_clean = _normalize_spaces(snippet)
        if snippet_clean:
            state.segments.append(snippet_clean)
            state.output_text = _join_segments(state.segments)
            return state.output_text, "append"
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

    segment = _dedupe_overlap(state.output_text, segment)
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
    
    normalized_output = _apply_personal_dictionary(updated or "")
    normalized_output = _polish_punctuation(_cleanup_disfluencies(normalized_output))
    
    if normalized_output != prev_output:
        return normalized_output, "append"
    
    return prev_output, "ignore"


def _llm_updated_transcript(
    prev_output: str,
    utterance: str,
    tone: str,
    language: str,
    intent: Optional[dict] = None,
) -> Optional[Tuple[str, str]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq
    except Exception:
        return None

    model = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=api_key)

    tone = (tone or "polite").strip().lower()
    language = (language or "english").strip().lower()

    system_prompt = (
        "You are DictaPilot Smart Editor - a precise voice dictation assistant. "
        "Your goal is to convert raw speech into clean, polished text while respecting user commands. "
        "Return JSON only with keys: updated_transcript, action. "
        "Valid actions: append, undo, undo_append, clear, ignore. "
        "Core rules:"
        "- Remove filler words (uh, um, you know, i mean) and repeated words"
        "- Apply proper punctuation (capitalization, periods, commas)"
        "- Preserve names, technical terms, and important content"
        f"- Default tone: {tone}. Language: {language}."
        "- Handle self-corrections intelligently (e.g., 'no not X use Y')"
        "- For undo commands, remove the last added content"
        "- For clear commands, return empty transcript with action 'clear'"
        "- For ignore commands, keep transcript unchanged"
        "- For 'X ignore' or 'ignore X' patterns, discard X and keep transcript unchanged"
    )
    if intent and intent.get("type") in {"rewrite", "grammar"}:
        mode = intent.get("type")
        target_tone = (intent.get("tone") or tone).strip().lower()
        user_prompt = (
            f"Current transcript:\n---\n{prev_output}\n---\n\n"
            f"Command: {utterance}\n"
            f"Task: {'Fix grammar/spelling and polish' if mode == 'grammar' else 'Rewrite'}\n"
            f"Tone: {target_tone}\n"
            f"Language: {language}\n"
            "Rewrite the FULL transcript accordingly and return JSON with updated_transcript and action.\n"
        )
    else:
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
            "   - 'replace X with Y': Replace X with Y in transcript\n"
            "   - rewrite/rephrase/polish: Rewrite the transcript with the requested tone\n"
            "   - fix grammar/spelling: Correct the transcript\n\n"
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


def smart_update_state(
    state: TranscriptState,
    utterance: str,
    mode: str = "heuristic",
    allow_llm: bool = True,
) -> Tuple[str, str, str]:
    selected_mode = (mode or "heuristic").strip().lower()
    if selected_mode not in {"heuristic", "llm"}:
        selected_mode = "heuristic"

    with state.lock:
        prev_output = state.output_text
        utterance = _apply_personal_dictionary(utterance or "")
        ctx = get_context()

        setting_cmd = _detect_setting_command(utterance)
        if setting_cmd:
            if setting_cmd["type"] == "tone":
                update_profile(ctx.app_id, tone=setting_cmd["value"])
            if setting_cmd["type"] == "language":
                update_profile(ctx.app_id, language=setting_cmd["value"])
            return prev_output, prev_output, "ignore"

        snippet = _resolve_snippet(utterance)
        if snippet:
            snippet_clean = _normalize_spaces(snippet)
            if snippet_clean:
                state.segments.append(snippet_clean)
                state.output_text = _join_segments(state.segments)
                return prev_output, state.output_text, "append"
            return prev_output, prev_output, "ignore"

        transform_cmd = _detect_transform_command(utterance)
        if transform_cmd and prev_output:
            llm_result = _llm_updated_transcript(prev_output, utterance, ctx.tone, ctx.language, transform_cmd)
            if llm_result is not None:
                new_output, action = llm_result
                if action == "ignore":
                    return prev_output, prev_output, "ignore"
                _sync_segments_from_output(state, prev_output, new_output)
                state.output_text = new_output
                return prev_output, state.output_text, action

        llm_always_clean = _env_flag("LLM_ALWAYS_CLEAN", "1")
        if not allow_llm:
            use_llm = False
        elif DICTATION_MODE == "speed":
            use_llm = False
        elif DICTATION_MODE == "accurate":
            use_llm = selected_mode == "llm"
        else:
            use_llm = selected_mode == "llm" and (llm_always_clean or needs_intent_handling(utterance))
        if use_llm:
            llm_result = _llm_updated_transcript(prev_output, utterance, ctx.tone, ctx.language, None)
            if llm_result is not None:
                new_output, action = llm_result
                if action == "ignore":
                    return prev_output, prev_output, "ignore"
                _sync_segments_from_output(state, prev_output, new_output)
                state.output_text = new_output
                return prev_output, state.output_text, action

        new_output, action = apply_heuristic(state, utterance)
        return prev_output, new_output, action


def sync_state_to_output(state: TranscriptState, prev_output: str, new_output: str) -> None:
    with state.lock:
        _sync_segments_from_output(state, prev_output, new_output)
        state.output_text = new_output
