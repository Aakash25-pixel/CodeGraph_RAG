from typing import List
from sentence_transformers import SentenceTransformer
from utils.config import EMBEDDING_MODEL_NAME
from utils.logger import get_logger

logger = get_logger(__name__)

class LocalEmbedder:
    """Generates embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            return []
            
        # sentence-transformers returns numpy arrays, we convert to lists
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
        
    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string."""
        return self.embed_texts([query])[0]
