import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple
from utils.models import Chunk
from embeddings.embedder import LocalEmbedder
from utils.logger import get_logger

logger = get_logger(__name__)

class FaissVectorStore:
    """Manages the FAISS index and ID mappings for semantic search."""
    
    def __init__(self, embedder: LocalEmbedder, index_path: str = None):
        self.embedder = embedder
        self.index_path = index_path
        self.mapping_path = f"{index_path}_mapping.json"
        
        self.index: faiss.Index = None
        self.id_to_chunk_id: Dict[int, str] = {}
        self.chunk_id_to_chunk: Dict[str, Chunk] = {}
        
    def _init_index(self, dimension: int):
        """Initializes a flat L2 index."""
        self.index = faiss.IndexFlatL2(dimension)
        logger.info(f"Initialized FAISS IndexFlatL2 with dimension {dimension}")

    def add_chunks(self, chunks: List[Chunk]):
        """Embeds and adds chunks to the index."""
        if not chunks:
            return
            
        logger.info(f"Embedding {len(chunks)} chunks...")
        texts = [chunk.text for chunk in chunks]
        embeddings = np.array(self.embedder.embed_texts(texts), dtype=np.float32)
        
        if self.index is None:
            self._init_index(embeddings.shape[1])
            
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        for i, chunk in enumerate(chunks):
            faiss_id = start_id + i
            self.id_to_chunk_id[faiss_id] = chunk.id
            self.chunk_id_to_chunk[chunk.id] = chunk
            
        logger.info(f"Added {len(chunks)} vectors to FAISS index. Total: {self.index.ntotal}")

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Searches the index for the most similar chunks.
        
        Args:
            query: The search query.
            k: Number of results to return.
            
        Returns:
            List of tuples (Chunk, distance). Lower distance is better (L2).
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on empty or uninitialized index.")
            return []
            
        query_emb = np.array([self.embedder.embed_query(query)], dtype=np.float32)
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in self.id_to_chunk_id:
                chunk_id = self.id_to_chunk_id[idx]
                if chunk_id in self.chunk_id_to_chunk:
                    results.append((self.chunk_id_to_chunk[chunk_id], float(dist)))
                    
        return results

    def save(self, index_path: str = None):
        """Saves the FAISS index and mappings to disk."""
        if self.index is None:
            return
            
        index_path = index_path or self.index_path
        mapping_path = f"{index_path}_mapping.json"
            
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.id_to_chunk_id, f)
            
        logger.info(f"Saved FAISS index to {index_path}")

    def load(self, chunks: List[Chunk], index_path: str = None):
        """
        Loads the FAISS index and mappings from disk.
        Needs the original chunks to reconstruct chunk_id_to_chunk mapping.
        """
        index_path = index_path or self.index_path
        mapping_path = f"{index_path}_mapping.json"
        
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            logger.warning("Index or mapping file not found. Starting fresh.")
            return
            
        self.index = faiss.read_index(index_path)
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            # json keys are always strings, need to convert back to int
            mapping_str_keys = json.load(f)
            self.id_to_chunk_id = {int(k): v for k, v in mapping_str_keys.items()}
            
        self.chunk_id_to_chunk = {chunk.id: chunk for chunk in chunks}
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
