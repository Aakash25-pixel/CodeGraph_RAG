from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from prompts.templates import REWRITER_PROMPT
from utils.models import RewriteDecision
from utils.llm import get_llm

class RewriterChain:
    """Rewrites user queries for better retrieval."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=RewriteDecision)
        
        template = REWRITER_PROMPT + "\n\n{format_instructions}"
        self.prompt = ChatPromptTemplate.from_template(template)
        
        self.chain = self.prompt | self.llm | self.parser
        
    def invoke(self, question: str) -> str:
        decision = self.chain.invoke({
            "question": question,
            "format_instructions": self.parser.get_format_instructions()
        })
        return decision.rewritten_query
