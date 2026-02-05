"""
DictaPilot Transcription Store
Handles persistent storage of all transcriptions in JSON format for easy access and reuse.

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
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


def get_transcriptions_path() -> Path:
    """Get platform-specific transcriptions file path"""
    system = platform.system()
    
    if system == "Windows":
        data_dir = Path(os.environ.get("APPDATA", ""))
        return data_dir / "DictaPilot" / "transcriptions.json"
    else:
        data_dir = Path(os.path.expanduser("~/.local/share"))
        return data_dir / "dictapilot" / "transcriptions.json"


def get_transcriptions_dir() -> Path:
    """Create and return transcriptions directory"""
    transcriptions_path = get_transcriptions_path()
    transcriptions_path.parent.mkdir(parents=True, exist_ok=True)
    return transcriptions_path.parent


@dataclass
class TranscriptionEntry:
    """Single transcription entry"""
    id: str
    timestamp: str
    timestamp_unix: float
    original_text: str
    processed_text: str
    action: str
    session_id: str
    word_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionEntry":
        return cls(**data)
    
    @property
    def display_text(self) -> str:
        """Get the best text for display/reuse (processed if available)"""
        return self.processed_text if self.processed_text else self.original_text


@dataclass
class TranscriptionStore:
    """Manages transcription storage and retrieval"""
    entries: List[TranscriptionEntry]
    session_start: str
    
    def __init__(self):
        self.entries = []
        self.session_start = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "session_start": self.session_start,
            "last_updated": datetime.now().isoformat(),
            "total_entries": len(self.entries),
            "entries": [e.to_dict() for e in self.entries]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionStore":
        store = cls()
        store.session_start = data.get("session_start", datetime.now().isoformat())
        if "entries" in data:
            store.entries = [TranscriptionEntry.from_dict(e) for e in data["entries"]]
        return store
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save all transcriptions to JSON file"""
        path = path or get_transcriptions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TranscriptionStore":
        """Load transcriptions from JSON file"""
        path = path or get_transcriptions_path()
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return cls()
    
    def add_entry(
        self,
        original_text: str,
        processed_text: str = "",
        action: str = "append"
    ) -> TranscriptionEntry:
        """Add a new transcription entry"""
        import uuid
        
        timestamp = datetime.now()
        entry = TranscriptionEntry(
            id=str(uuid.uuid4())[:8],
            timestamp=timestamp.isoformat(),
            timestamp_unix=timestamp.timestamp(),
            original_text=original_text,
            processed_text=processed_text,
            action=action,
            session_id=self.session_start,
            word_count=len(original_text.split())
        )
        self.entries.append(entry)
        return entry
    
    def get_recent(self, count: int = 10) -> List[TranscriptionEntry]:
        """Get the most recent transcriptions"""
        return self.entries[-count:] if count > 0 else self.entries
    
    def get_all(self) -> List[TranscriptionEntry]:
        """Get all transcriptions"""
        return self.entries
    
    def search(self, query: str, case_sensitive: bool = False) -> List[TranscriptionEntry]:
        """Search transcriptions by text content"""
        if not query:
            return self.entries
        
        if case_sensitive:
            return [e for e in self.entries 
                    if query in e.original_text or query in e.processed_text]
        else:
            query_lower = query.lower()
            return [e for e in self.entries 
                    if query_lower in e.original_text.lower() or 
                       query_lower in e.processed_text.lower()]
    
    def get_by_date(self, date_str: str) -> List[TranscriptionEntry]:
        """Get transcriptions from a specific date (YYYY-MM-DD)"""
        return [e for e in self.entries if e.timestamp.startswith(date_str)]
    
    def clear_session(self) -> None:
        """Clear all entries and start a new session"""
        self.entries = []
        self.session_start = datetime.now().isoformat()
    
    def export_to_text(self, path: Optional[Path] = None, 
                       include_metadata: bool = False) -> str:
        """Export all transcriptions to a plain text format"""
        lines = []
        for entry in self.entries:
            if include_metadata:
                lines.append(f"[{entry.timestamp}]")
            lines.append(entry.display_text)
            lines.append("")  # Empty line between entries
        text = "\n".join(lines).strip()
        
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text
    
    def export_to_json(self, path: Optional[Path] = None) -> str:
        """Export to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_words = sum(e.word_count for e in self.entries)
        total_chars = sum(len(e.display_text) for e in self.entries)
        actions = {}
        for e in self.entries:
            actions[e.action] = actions.get(e.action, 0) + 1
        
        return {
            "total_transcriptions": len(self.entries),
            "total_words": total_words,
            "total_characters": total_chars,
            "action_breakdown": actions,
            "session_start": self.session_start
        }


# Global store instance
_store: Optional[TranscriptionStore] = None


def get_store() -> TranscriptionStore:
    """Get or create the global transcription store"""
    global _store
    if _store is None:
        _store = TranscriptionStore.load()
    return _store


def save_store() -> None:
    """Save the global store to disk"""
    global _store
    if _store is not None:
        _store.save()


def add_transcription(original_text: str, processed_text: str = "", 
                      action: str = "append") -> TranscriptionEntry:
    """Convenience function to add a transcription"""
    store = get_store()
    entry = store.add_entry(original_text, processed_text, action)
    save_store()
    return entry


def get_transcriptions(count: int = 10) -> List[TranscriptionEntry]:
    """Get recent transcriptions"""
    return get_store().get_recent(count)


def search_transcriptions(query: str) -> List[TranscriptionEntry]:
    """Search transcriptions"""
    return get_store().search(query)


def export_all_to_text(path: Optional[Path] = None, 
                       include_metadata: bool = False) -> str:
    """Export all transcriptions to text file"""
    return get_store().export_to_text(path, include_metadata)


def clear_all_transcriptions() -> None:
    """Clear all transcriptions"""
    store = get_store()
    store.clear_session()
    save_store()


def get_storage_info() -> Dict[str, Any]:
    """Get storage location and statistics"""
    store = get_store()
    return {
        "storage_path": str(get_transcriptions_path()),
        "statistics": store.get_statistics()
    }


if __name__ == "__main__":
    # Demo/test of the transcription store
    store = get_store()
    print(f"Transcription Store Demo")
    print(f"Storage location: {get_transcriptions_path()}")
    print(f"Current entries: {len(store.entries)}")
    
    # Add a demo entry
    add_transcription("Hello world", "Hello world", "append")
    print(f"After adding demo: {len(store.entries)} entries")
    
    # Show statistics
    stats = store.get_statistics()
    print(f"Stats: {stats}")
