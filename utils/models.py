from pydantic import BaseModel, Field
from typing import List, Optional

class ChunkMetadata(BaseModel):
    filepath: str
    symbol_name: str
    symbol_type: str  # 'function', 'class', 'method'
    start_line: int
    end_line: int
    parent_class: Optional[str] = None
    docstring: Optional[str] = None

class Chunk(BaseModel):
    id: str  # Unique ID, e.g., filepath:symbol_name
    text: str # The actual source code
    metadata: ChunkMetadata

# For structured output in LangChain
class RouterDecision(BaseModel):
    """Decision on which retrieval method to use."""
    retrieval_type: str = Field(description="Must be one of: 'VECTOR', 'GRAPH', or 'HYBRID'")
    reasoning: str = Field(description="Why this retrieval type was chosen")

class GraderDecision(BaseModel):
    """Decision on whether the retrieved context is sufficient."""
    is_sufficient: bool = Field(description="True if the context provides enough information to answer the question")
    reasoning: str = Field(description="Why the context is or isn't sufficient")

class RewriteDecision(BaseModel):
    """Rewritten query for better retrieval."""
    rewritten_query: str = Field(description="The reformulated question")
