import hashlib
import os

from ollama import Ai
from chunker import parse, chunk, get_language_config
from chroma import upsert_chunk, delete_file, get_file_hash


def index(path):
    """Index a file into ChromaDB. Skips if file content hasn't changed.
    Returns a status dict with 'status' and 'message' keys."""

    if not os.path.isfile(path):
        return {"status": "error", "message": f"File not found: {path}"}

    if get_language_config(path) is None:
        return {"status": "error", "message": f"Unsupported file type: {os.path.splitext(path)[1]}"}

    # Hash file contents to detect changes
    with open(path, "rb") as f:
        content = f.read()
    content_hash = hashlib.sha256(content).hexdigest()

    # Skip re-indexing if file hasn't changed
    stored_hash = get_file_hash(path)
    if stored_hash == content_hash:
        return {"status": "skipped", "message": "File unchanged, using cached index."}

    # Clear old chunks before re-indexing
    if stored_hash is not None:
        delete_file(path)

    tree, source, config = parse(path)
    chunks = chunk(tree, source, config)

    if not chunks:
        return {"status": "warning", "message": "No semantic chunks found in file."}

    # Embed each chunk and store in ChromaDB
    for c in chunks:
        embedding = Ai(c["text"]).embed()
        chunk_id = hashlib.sha256(f"{path}:{c['start_line']}:{c['end_line']}".encode()).hexdigest()
        metadata = {
            "path": path,
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "type": c["type"],
            "content_hash": content_hash,
        }
        upsert_chunk(chunk_id, embedding, c["text"], metadata)

    return {"status": "indexed", "message": f"Indexed {len(chunks)} chunks."}
