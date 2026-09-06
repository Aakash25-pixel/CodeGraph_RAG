# CodeGraph RAG

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit)](https://codegraph.streamlit.app/)

CodeGraph RAG is a powerful, Polyglot Retrieval-Augmented Generation (RAG) tool designed to understand, index, and query software repositories intelligently. Instead of relying solely on text-based vector search, it constructs a structural Knowledge Graph of your codebase (mapping file hierarchies, function calls, and class inheritances) and combines it with FAISS vector embeddings to give an LLM profound architectural context.

## 🌟 Features

- **Polyglot Parsing**: Built on Tree-sitter, it natively understands Python (`.py`), JavaScript (`.js`, `.jsx`), TypeScript (`.ts`, `.tsx`), HTML, and CSS.
- **Hybrid Retrieval (Vector + Graph)**: Finds semantically relevant entry points using FAISS, and then traverses a NetworkX Knowledge Graph to pull in relevant caller/callee context.
- **Mind-Map Visualization**: Includes an interactive, Force-Directed Graphviz UI inside Streamlit to visualize your codebase architecture in real-time.
- **Multi-Repo Isolation**: Safely index multiple repositories simultaneously. Artifacts are partitioned safely using MD5 URL hashing.
- **Optimized for Open-Source LLMs**: Configured out of the box to connect to HuggingFace Serverless endpoints (defaulting to `Qwen/Qwen2.5-Coder-32B-Instruct`) with robust exponential backoff.

## 🚀 Getting Started

### Prerequisites
- Python 3.13 (or compatible)
- A HuggingFace account/token

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Aakash25-pixel/CodeGraph_RAG.git
   cd CodeGraph_RAG
   ```
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your HuggingFace token:
   ```env
   HF_TOKEN=your_huggingface_token_here
   USE_HUGGINGFACE=true
   LLM_MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct
   ```

### Running the Application
Launch the Streamlit UI:
```bash
streamlit run app/main.py
```
From the sidebar, enter any public GitHub URL or local path, hit **Index Repository**, and start chatting!

## 🧪 Testing
Integration tests are provided using `pytest` to ensure accurate AST chunking, correct JS extraction, and timeout resilience.
```bash
python -m pytest tests/test_e2e.py -v
```

## 📖 Documentation
For a deep dive into the architecture, AST chunking strategies, and the Hybrid Retrieval Pipeline, please read [DOCUMENTATION.md](DOCUMENTATION.md).
