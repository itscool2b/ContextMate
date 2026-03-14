from fastmcp import FastMCP
from ollama import Ai
from chroma import query_chunks
from indexingpipeline import index
from watcher import scan

mcp = FastMCP(
    name="context-mate",
    instructions="Semantic code search MCP server. Indexes code into meaningful chunks and returns only what's relevant. Supports Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, and C#.",
)

session_stats = {"queries": 0, "chunks_returned": 0}


@mcp.tool
def get_session_summary():
    """Returns stats on how many queries and chunks were served this session."""
    return {
        "queries": session_stats["queries"],
        "chunks_returned": session_stats["chunks_returned"],
        "message": f"Served {session_stats['chunks_returned']} targeted chunks across {session_stats['queries']} queries instead of dumping entire files.",
    }


@mcp.tool
def read_file(path: str, reason: str):
    """Index a file and return only the chunks relevant to the given reason.
    Supports: .py .js .jsx .mjs .ts .tsx .rs .go .java .c .h .cpp .cc .cxx .hpp .rb .cs"""
    result = index(path)
    if result["status"] == "error":
        return {"error": result["message"]}

    query_embedding = Ai(reason).embed()
    results = query_chunks(query_embedding, path)
    session_stats["queries"] += 1
    session_stats["chunks_returned"] += len(results["documents"][0])
    return {
        "path": path,
        "reason": reason,
        "index_status": result["message"],
        "chunks": [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ],
    }


@mcp.tool
def search_codebase(query: str):
    """Search all indexed files for code matching a natural language query."""
    query_embedding = Ai(query).embed()
    results = query_chunks(query_embedding, path=None, n_results=10)
    session_stats["queries"] += 1
    session_stats["chunks_returned"] += len(results["documents"][0])
    return {
        "query": query,
        "results": [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ],
    }


@mcp.tool
def index_directory(path: str):
    """Index all supported files in a directory."""
    results = scan(path)
    return {"indexed": sum(1 for r in results if r["status"] == "indexed"), "total": len(results)}


if __name__ == "__main__":
    mcp.run()
