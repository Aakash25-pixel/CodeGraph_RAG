import os
import pytest
import shutil
import tempfile
from unittest.mock import patch, MagicMock

from ingestion.chunker import RepoChunker
from utils.config import get_repo_storage_dir
from retrieval.orchestrator import CodeGraphOrchestrator
from graph.traversal import GraphTraverser
from embeddings.vectorstore import FaissVectorStore
from embeddings.embedder import LocalEmbedder

# Fixtures for creating mock repositories
@pytest.fixture
def mock_repo_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_empty_repository(mock_repo_dir):
    """Test that an empty repository yields 0 chunks and handles gracefully."""
    chunker = RepoChunker(mock_repo_dir)
    chunks = chunker.chunk_repo()
    assert len(chunks) == 0

def test_python_ml_script(mock_repo_dir):
    """Test that top-level ML logic is captured in a Python file."""
    py_code = """
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def load_data():
    return np.array([1, 2, 3])

# Module level pipeline definition
model = RandomForestClassifier()
model.fit([[0], [1]], [0, 1])
    """
    
    file_path = os.path.join(mock_repo_dir, "train.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(py_code)
        
    chunker = RepoChunker(mock_repo_dir)
    chunks = chunker.chunk_repo()
    
    assert len(chunks) > 0
    
    module_chunk = next((c for c in chunks if "<module_level>" in c.metadata.symbol_name), None)
    assert module_chunk is not None
    assert "RandomForestClassifier" in module_chunk.text
    
    func_chunk = next((c for c in chunks if c.metadata.symbol_name == "load_data"), None)
    assert func_chunk is not None

def test_js_react_arrow_component(mock_repo_dir):
    """Test that JS arrow functions are correctly named from variable declarations."""
    js_code = """
import React from 'react';

const MyComponent = () => {
    return <div>Hello</div>;
};

export default MyComponent;
    """
    
    file_path = os.path.join(mock_repo_dir, "App.jsx")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(js_code)
        
    chunker = RepoChunker(mock_repo_dir)
    chunks = chunker.chunk_repo()
    
    # Check that MyComponent is named correctly, not 'unknown'
    arrow_chunk = next((c for c in chunks if c.metadata.symbol_name == "MyComponent"), None)
    assert arrow_chunk is not None
    assert "<div>Hello</div>" in arrow_chunk.text

@patch("chains.synthesizer.SynthesizerChain.invoke")
def test_provider_timeout_handling(mock_invoke, mock_repo_dir):
    """Test that the orchestrator retries and handles timeouts gracefully."""
    from httpx import HTTPStatusError, Request
    from httpx import Response
    
    mock_invoke.side_effect = HTTPStatusError(
        "Timeout", 
        request=Request("POST", "http://test"), 
        response=Response(408, request=Request("POST", "http://test"))
    )
    
    embedder = LocalEmbedder()
    vectorstore = FaissVectorStore(embedder)
    # mock vector search
    vectorstore.search = MagicMock(return_value=[])
    
    traverser = GraphTraverser(MagicMock())
    orchestrator = CodeGraphOrchestrator(vectorstore, traverser)
    
    result = orchestrator.answer("What does this project do?")
    assert "timed out or failed" in result
