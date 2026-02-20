import os
import re
from typing import List, Dict, Any, Tuple

CHUNK_MIN = 800
CHUNK_MAX = 1200
CHUNK_OVERLAP = 200


def ensure_dirs():
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./vectorstore", exist_ok=True)


def sanitize_filename(name: str) -> str:
    # Keep alphanum, dash, underscore, dot
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name


def make_unique_path(dest_dir: str, base_name: str) -> str:
    base_name = sanitize_filename(base_name)
    full = os.path.join(dest_dir, base_name)
    if not os.path.exists(full):
        return full
    root, ext = os.path.splitext(base_name)
    i = 1
    while True:
        candidate = os.path.join(dest_dir, f"{root}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def char_chunk_text(text: str,
                     chunk_min: int = CHUNK_MIN,
                     chunk_max: int = CHUNK_MAX,
                     overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple greedy chunker by characters, attempts max size, respects overlap.
    """
    text = text.strip()
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_max, n)
        # Try to end at a sentence boundary near end
        window = text[start:end]
        # Find last sentence break within window
        m = re.search(r"[.!?]\s+[^\S\n]*$", window)
        if m and (len(window) >= chunk_min):
            end = start + m.end()
            chunk = text[start:end]
        else:
            chunk = window
        chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return [c.strip() for c in chunks if c.strip()]


def build_chunk_metadata(source: str, page: int | None, chunk_id: int) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "source": os.path.basename(source),
        "chunk_id": f"chunk_{chunk_id}",
    }
    if page is not None:
        meta["page"] = page
    return meta


def format_citation(meta: Dict[str, Any]) -> str:
    source = meta.get("source", "unknown")
    page = meta.get("page")
    chunk_id = meta.get("chunk_id", "chunk_?")
    if page is not None:
        return f"[{source} p.{page} {chunk_id}]"
    return f"[{source} {chunk_id}]"


def join_citations(metadatas: List[Dict[str, Any]]) -> str:
    uniq: List[str] = []
    seen = set()
    for m in metadatas:
        c = format_citation(m)
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return " ".join(uniq)


def guard_unrelated(question: str) -> bool:
    """More permissive: only block clearly off-topic requests, allow greetings and general chat."""
    q = question.lower().strip()
    
    # Allow common greetings and polite conversation
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you', 'thanks', 'thank you']
    if any(greeting in q for greeting in greetings):
        return False
    
    # Allow questions about the assistant itself
    if any(phrase in q for phrase in ['what can you do', 'what are you', 'who are you', 'help']):
        return False
    
    # Only block clearly unrelated topics
    unrelated_patterns = [
        r'\b(weather|news|sports|movies?|music|songs?)\b',
        r'\b(recipe|cooking|food)\b',
        r'\b(travel|vacation|hotel)\b',
        r'\bwrite.*story\b',
        r'\bplay.*game\b',
        r'\btell.*joke\b'
    ]
    return any(re.search(p, q) for p in unrelated_patterns)


def low_confidence(retrieved: List[Tuple[str, Dict[str, Any]]]) -> bool:
    """Very simple: if fewer than 2 chunks or total chars < 800, mark low confidence."""
    if len(retrieved) < 1:
        return True
    total = sum(len(t[0]) for t in retrieved)
    return total < 800
