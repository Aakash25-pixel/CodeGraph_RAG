import streamlit as st
import os
import sys

# Add the project root to sys.path so modules can be imported when running `streamlit run app/main.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.fetcher import RepoFetcher
from ingestion.chunker import RepoChunker
from embeddings.embedder import LocalEmbedder
from embeddings.vectorstore import FaissVectorStore
from graph.builder import GraphBuilder
from graph.traversal import GraphTraverser
from retrieval.orchestrator import CodeGraphOrchestrator
from utils.logger import get_logger
from utils.config import get_repo_storage_dir

logger = get_logger(__name__)

st.set_page_config(page_title="CodeGraph", page_icon="🕸️", layout="wide")

def init_system(repo_url: str):
    """Initializes the backend components by indexing the repository."""
    st.session_state.indexing = True
    
    with st.spinner(f"Fetching repository from {repo_url}..."):
        fetcher = RepoFetcher()
        local_path = fetcher.fetch(repo_url)
        
    with st.spinner("Parsing and chunking source files..."):
        chunker = RepoChunker(local_path)
        chunks = chunker.chunk_repo()
        
        if len(chunks) == 0:
            st.error("No valid code chunks found in this repository! Indexing aborted.")
            st.session_state.indexing = False
            return
            
        paths = get_repo_storage_dir(repo_url)
        chunker.save_chunks(paths['chunks'])
        st.success(f"Extracted {len(chunks)} chunks.")
        
    with st.spinner("Building vector index..."):
        embedder = LocalEmbedder()
        vectorstore = FaissVectorStore(embedder)
        vectorstore.add_chunks(chunks)
        vectorstore.save(paths['vector_index'])
        
    with st.spinner("Building knowledge graph..."):
        builder = GraphBuilder(local_path, chunks)
        builder.build()
        builder.save(paths['graph'])
        st.success(f"Built graph with {builder.graph.number_of_nodes()} nodes.")
        
    st.session_state.current_repo_url = repo_url
    st.session_state.system_ready = True
    st.session_state.indexing = False
    # Clear the LLM/vector cache since we indexed a new repo
    load_system.clear()

@st.cache_resource
def load_system(repo_url: str):
    """Loads pre-built indices from disk."""
    if not repo_url:
        return None
        
    paths = get_repo_storage_dir(repo_url)
    if not os.path.exists(paths['vector_index']) or not os.path.exists(paths['graph']):
        return None
        
    embedder = LocalEmbedder()
    vectorstore = FaissVectorStore(embedder)
    
    # We need to load chunks to pass to vectorstore
    chunker = RepoChunker(".")
    chunks = chunker.load_chunks(paths['chunks'])
    if not chunks:
        return None
        
    vectorstore.load(chunks, paths['vector_index'])
    graph = GraphBuilder.load(paths['graph'])
    traverser = GraphTraverser(graph)
    
    return CodeGraphOrchestrator(vectorstore, traverser)

# UI Layout
st.title("🕸️ CodeGraph RAG")
st.markdown("Ask natural language questions about your Python codebase.")

# Sidebar for indexing
with st.sidebar:
    st.header("Repository Configuration")
    repo_input = st.text_input("GitHub URL or Local Path", placeholder="https://github.com/user/repo")
    if st.button("Index Repository", disabled=st.session_state.get('indexing', False)):
        if repo_input:
            init_system(repo_input)
        else:
            st.warning("Please provide a repository URL or path.")
            
    if st.session_state.get('system_ready', False):
        st.success("System is ready!")
        
        with st.expander("🕸️ Visualize Knowledge Graph"):
            if st.button("Generate Graph Diagram"):
                with st.spinner("Generating diagram..."):
                    orchestrator = load_system(st.session_state.get('current_repo_url', ''))
                    if orchestrator and orchestrator.graph.graph.number_of_nodes() > 0:
                        nx_graph = orchestrator.graph.graph
                        
                        # Use 'fdp' for force-directed placement (optimizes 2D space) or 'twopi' for radial
                        dot_str = 'digraph {\n  layout=fdp;\n  overlap=false;\n  splines=true;\n  K=0.8;\n'
                        dot_str += '  node [fontname="Segoe UI, Arial", fontsize=11, margin=0.2];\n'
                        dot_str += '  edge [color="gray50", penwidth=1.2];\n'
                        
                        # Add all nodes explicitly
                        for node, data in nx_graph.nodes(data=True):
                            node_id = str(node).replace('"', "'")
                            meta = data.get("metadata", {})
                            
                            if node_id.startswith("FILE:"):
                                filename = node_id.replace("FILE:", "").split("/")[-1].split("\\\\")[-1]
                                label = f"📁 {filename}"
                                dot_str += f'  "{node_id}" [label="{label}", shape="note", style="filled", fillcolor="#fff3cd"];\n'
                            else:
                                symbol_name = meta.get("symbol_name", "") if isinstance(meta, dict) else ""
                                symbol_type = meta.get("symbol_type", "") if isinstance(meta, dict) else ""
                                
                                # Friendly naming
                                if symbol_name.startswith("<module_level>"):
                                    if "_part" in symbol_name:
                                        part_num = symbol_name.split("_part")[1]
                                        label = f"🌐 Global Logic (Part {part_num})"
                                    else:
                                        label = "🌐 Global Logic"
                                    fillcolor = "#e2e3e5"
                                    shape = "box"
                                elif symbol_type == "class":
                                    label = f"📦 {symbol_name}"
                                    fillcolor = "#d1e7dd"
                                    shape = "component"
                                else:
                                    label = f"⚙️ {symbol_name}"
                                    fillcolor = "#cfe2ff"
                                    shape = "box"
                                    
                                dot_str += f'  "{node_id}" [label="{label}", shape="{shape}", style="filled,rounded", fillcolor="{fillcolor}"];\n'
                            
                        # Add edges
                        for u, v, data in nx_graph.edges(data=True):
                            edge_type = data.get('type', '')
                            # Make relations friendly
                            friendly_edge = "contains" if edge_type == "CONTAINS" else "calls" if edge_type == "CALLS" else "inherits" if edge_type == "INHERITS" else edge_type
                            u_id = str(u).replace('"', "'")
                            v_id = str(v).replace('"', "'")
                            # Add dashed style for Contains to separate structure from logic
                            style = 'style="dashed", color="gray60"' if edge_type == "CONTAINS" else 'color="#0d6efd", penwidth=1.5'
                            dot_str += f'  "{u_id}" -> "{v_id}" [label=" {friendly_edge} ", fontsize=9, fontcolor="gray30", {style}];\n'
                            
                        dot_str += "}\n"
                        st.graphviz_chart(dot_str)
                    else:
                        st.info("Graph is empty or not loaded.")
    else:
        st.info("Please index a repository to begin.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the repository...", disabled=not st.session_state.get('system_ready', False)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            orchestrator = load_system(st.session_state.get('current_repo_url', ''))
            if orchestrator:
                try:
                    answer = orchestrator.answer(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    logger.exception(e)
            else:
                st.error("System failed to load. Please re-index the repository.")
