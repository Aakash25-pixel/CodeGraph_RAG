from typing import List, Set
from utils.models import Chunk
from embeddings.vectorstore import FaissVectorStore
from graph.traversal import GraphTraverser
from chains.synthesizer import SynthesizerChain
from utils.logger import get_logger
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from httpx import HTTPStatusError

logger = get_logger(__name__)

class CodeGraphOrchestrator:
    """Orchestrates the entire CodeGraph retrieval and answer generation loop."""
    
    def __init__(self, vectorstore: FaissVectorStore, graph_traverser: GraphTraverser):
        self.vectorstore = vectorstore
        self.graph = graph_traverser
        self.synthesizer = SynthesizerChain()
        
    def _format_context(self, chunks: List[Chunk]) -> str:
        """Formats chunks into a string for the LLM."""
        formatted = []
        for c in chunks:
            formatted.append(f"--- {c.metadata.filepath}:{c.metadata.start_line}-{c.metadata.end_line} [{c.metadata.symbol_type} {c.metadata.symbol_name}] ---\n{c.text}\n")
        return "\n".join(formatted)
        
    def _retrieve(self, query: str, strategy: str = 'HYBRID') -> List[Chunk]:
        """Retrieves chunks based on the strategy."""
        retrieved_dict = {}
        
        # Always do a vector search as a baseline to find entry points
        vector_results = self.vectorstore.search(query, k=5)
        for chunk, _ in vector_results:
            retrieved_dict[chunk.id] = chunk
            
        if strategy in ['GRAPH', 'HYBRID']:
            # Expand context using graph traversal from the vector hits
            expanded_ids = set()
            for chunk_id in list(retrieved_dict.keys()):
                # get context subgraph (callers, callees, etc. up to depth 1)
                neighbors = self.graph.get_context_subgraph(chunk_id, depth=1)
                for neighbor_id in neighbors:
                    if neighbor_id in self.vectorstore.chunk_id_to_chunk:
                        expanded_ids.add(neighbor_id)
            for n_id in expanded_ids:
                retrieved_dict[n_id] = self.vectorstore.chunk_id_to_chunk[n_id]
            
        return list(retrieved_dict.values())

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(2)
    )
    def _invoke_llm_with_retry(self, question: str, context: str) -> str:
        return self.synthesizer.invoke(question, context)

    def answer(self, original_question: str) -> str:
        """Runs the retrieval loop to answer the question."""
        logger.info(f"Question: {original_question}")
        
        # 1. Direct Retrieval
        context_chunks = self._retrieve(original_question, strategy='HYBRID')
        final_context = self._format_context(context_chunks)
        
        # 2. Synthesize with retry
        try:
            answer = self._invoke_llm_with_retry(original_question, final_context)
            return answer
        except Exception as e:
            logger.error(f"LLM failure: {e}")
            return f"Model provider timed out or failed; try again. Details: {str(e)}"
