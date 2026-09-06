# CodeGraph RAG Architecture & Documentation

## Overview
CodeGraph RAG is engineered to solve a fundamental limitation in standard Code RAG systems: **loss of structural context**. When code is blindly split into text chunks, the LLM cannot understand how a modified function affects its callers, or what base class an object inherits from. 

To solve this, CodeGraph parses the Abstract Syntax Tree (AST) of the repository, extracting discrete logical units (Classes, Functions, Modules), embedding them for semantic search, and interlinking them using a Knowledge Graph.

## Core Modules

### 1. `ingestion.parser` (Polyglot AST Parser)
The ingestion engine relies on `tree-sitter` bindings for multiple languages.
- **Language Map**: Dynamically loads the grammar based on file extension (Python, JavaScript, TypeScript, HTML, CSS).
- **Execution**: Converts the raw source code into an AST ready for querying.

### 2. `ingestion.chunker` (Intelligent Chunking)
Standard text splitters are not used. Instead, the AST is traversed to create **Logical Chunks**:
- **Functions & Methods**: Identifies `function_definition` (Python), `arrow_function` (JS/React), and `method_definition`.
- **Classes**: Extracts `class_definition` nodes and recursively parses their children.
- **Global Context (`<module_level>`)**: Extracts top-level imports, pipeline definitions, and global variables. If a module is massively large, it automatically slices it into manageable 150-line sub-chunks to protect the LLM context window.

### 3. `embeddings.vectorstore` (Semantic Search)
- **Model**: `all-MiniLM-L6-v2` via `sentence-transformers`.
- **Backend**: Uses FAISS (`IndexFlatL2`) for high-speed, local vector similarity search. 
- **Storage**: Maps integer FAISS IDs to persistent string Chunk IDs, serializing everything to a localized, MD5-hashed folder unique to the repository URL.

### 4. `graph.builder` & `graph.traversal` (Knowledge Graph)
- **Node Construction**: Creates explicit graph nodes for `Chunks` and `Files`.
- **Edge Extraction**: Analyzes the AST for `call_expression` or `inheritance` nodes to draw relationships (`CALLS`, `INHERITS`). Draws structural boundaries using `CONTAINS` edges from Files down to their respective code blocks.
- **Traversal**: Given a starting chunk ID, `get_context_subgraph(depth=1)` fetches the immediate structural neighborhood (e.g., retrieving the caller of the target function).

### 5. `retrieval.orchestrator` (The Hybrid Pipeline)
Orchestrates the entire query process:
1. **Vector Search**: Identifies the Top-K (default=5) chunks semantically matching the user's question.
2. **Graph Expansion**: For each vector hit, it traverses the Knowledge Graph to pull in adjacent nodes (e.g., pulling in a class definition if a hit occurred inside one of its methods).
3. **Synthesis**: Concatenates the hybrid context array into a well-formatted string.
4. **Resilient LLM Invocation**: Uses `@retry` from the `tenacity` library with exponential backoff to handle free-tier API rate limits or cold-start timeouts gracefully (max 120s timeout, up to 2 attempts).

### 6. `app.main` (Streamlit Dashboard)
- **State Management**: Uses `@st.cache_resource` to keep the LLM client, VectorStore, and Graph Traverser in memory across fast UI rerenders.
- **Graphviz Visualization**: Dynamically constructs a `dot` string to render a beautiful, Force-Directed Placement (`fdp`) layout of the codebase. It substitutes coder-centric backend IDs for user-friendly, emoji-driven labels (e.g., `📁`, `⚙️`, `📦`, `🌐`).
