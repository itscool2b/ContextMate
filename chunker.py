import importlib
from pathlib import Path
from tree_sitter import Language, Parser

# Language registry: file extension -> parsing config
# Each config defines:
#   module        - tree-sitter grammar package name
#   top_level     - AST node types to extract as chunks
#   wrappers      - nodes that wrap a target (e.g. decorators, exports)
#   nested        - node types to also extract from inside class-like nodes (Java, C#)
#   ts_dialect    - only for tree-sitter-typescript which has two sub-languages

LANGUAGES = {
    ".py": {
        "module": "tree_sitter_python",
        "top_level": {"function_definition", "class_definition"},
        "wrappers": {"decorated_definition": {"function_definition", "class_definition"}},
    },
    ".js": {
        "module": "tree_sitter_javascript",
        "top_level": {"function_declaration", "class_declaration"},
        "wrappers": {"export_statement": {"function_declaration", "class_declaration"}},
    },
    ".ts": {
        "module": "tree_sitter_typescript",
        "ts_dialect": "typescript",
        "top_level": {
            "function_declaration", "class_declaration",
            "interface_declaration", "type_alias_declaration",
            "enum_declaration",
        },
        "wrappers": {"export_statement": {
            "function_declaration", "class_declaration",
            "interface_declaration", "type_alias_declaration",
            "enum_declaration",
        }},
    },
    ".rs": {
        "module": "tree_sitter_rust",
        "top_level": {"function_item", "struct_item", "enum_item", "trait_item", "impl_item"},
    },
    ".go": {
        "module": "tree_sitter_go",
        "top_level": {"function_declaration", "method_declaration", "type_declaration"},
    },
    ".java": {
        "module": "tree_sitter_java",
        "top_level": {"class_declaration", "interface_declaration", "enum_declaration"},
        "nested": {"method_declaration", "constructor_declaration"},
    },
    ".c": {
        "module": "tree_sitter_c",
        "top_level": {"function_definition", "struct_specifier", "enum_specifier"},
    },
    ".cpp": {
        "module": "tree_sitter_cpp",
        "top_level": {"function_definition", "class_specifier", "struct_specifier", "namespace_definition"},
    },
    ".rb": {
        "module": "tree_sitter_ruby",
        "top_level": {"method", "class", "module"},
    },
    ".cs": {
        "module": "tree_sitter_c_sharp",
        "top_level": {"class_declaration", "struct_declaration", "interface_declaration", "enum_declaration"},
        "nested": {"method_declaration", "constructor_declaration"},
    },
}

# Aliases for alternate extensions
LANGUAGES[".jsx"] = LANGUAGES[".js"]
LANGUAGES[".mjs"] = LANGUAGES[".js"]
LANGUAGES[".tsx"] = {
    **LANGUAGES[".ts"],
    "ts_dialect": "tsx",
}
LANGUAGES[".h"] = LANGUAGES[".c"]
LANGUAGES[".cc"] = LANGUAGES[".cpp"]
LANGUAGES[".cxx"] = LANGUAGES[".cpp"]
LANGUAGES[".hpp"] = LANGUAGES[".cpp"]

# Cache loaded Language objects to avoid re-importing
_lang_cache = {}
_parser = Parser()


def get_language_config(path):
    """Return language config for a file path, or None if unsupported."""
    ext = Path(path).suffix.lower()
    return LANGUAGES.get(ext)


def _load_language(config):
    """Load and cache a tree-sitter Language from its grammar package."""
    module_name = config["module"]
    cache_key = f"{module_name}:{config.get('ts_dialect', '')}"

    if cache_key not in _lang_cache:
        mod = importlib.import_module(module_name)
        # tree-sitter-typescript exposes language_typescript() and language_tsx()
        dialect = config.get("ts_dialect")
        if dialect == "tsx":
            _lang_cache[cache_key] = Language(mod.language_tsx())
        elif dialect == "typescript":
            _lang_cache[cache_key] = Language(mod.language_typescript())
        else:
            _lang_cache[cache_key] = Language(mod.language())

    return _lang_cache[cache_key]


def parse(path):
    """Parse a file with the correct tree-sitter grammar.
    Returns (tree, source_bytes, config). Raises ValueError if unsupported."""
    config = get_language_config(path)
    if config is None:
        raise ValueError(f"Unsupported file type: {Path(path).suffix}")

    lang = _load_language(config)
    _parser.language = lang

    with open(path, "rb") as f:
        source = f.read()

    return _parser.parse(source), source, config


def chunk(tree, source, config):
    """Extract semantic chunks from a parsed AST."""
    chunks = []
    top_level = config["top_level"]
    wrappers = config.get("wrappers", {})
    nested = config.get("nested", set())

    for node in tree.root_node.children:
        target = node

        # Unwrap wrapper nodes (decorators, export statements, etc.)
        if node.type in wrappers:
            inner_types = wrappers[node.type]
            for child in node.children:
                if child.type in inner_types:
                    target = child
                    break
            if target is node and node.type not in top_level:
                continue

        if target.type in top_level:
            chunks.append({
                "text": source[node.start_byte:node.end_byte].decode("utf-8"),
                "start_line": node.start_point[0],
                "end_line": node.end_point[0],
                "type": target.type,
            })

            # For Java/C#: also extract methods from inside classes
            if nested:
                _extract_nested(node, source, nested, chunks)

    return chunks


def _extract_nested(parent, source, target_types, chunks):
    """Pull out nested definitions (methods, constructors) from class-like nodes."""
    for child in parent.children:
        if child.type in target_types:
            chunks.append({
                "text": source[child.start_byte:child.end_byte].decode("utf-8"),
                "start_line": child.start_point[0],
                "end_line": child.end_point[0],
                "type": child.type,
            })
        # Recurse into body blocks to find methods
        elif child.type in ("class_body", "declaration_list", "block"):
            _extract_nested(child, source, target_types, chunks)
