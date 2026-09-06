from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from prompts.templates import ROUTER_PROMPT
from utils.models import RouterDecision
from utils.llm import get_llm

class RouterChain:
    """Classifies user questions into retrieval strategies."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=RouterDecision)
        
        # Append format instructions so HF/Ollama models know to output JSON
        template = ROUTER_PROMPT + "\n\n{format_instructions}"
        self.prompt = ChatPromptTemplate.from_template(template)
        
        self.chain = self.prompt | self.llm | self.parser
        
    def invoke(self, question: str) -> RouterDecision:
        return self.chain.invoke({
            "question": question,
            "format_instructions": self.parser.get_format_instructions()
        })
