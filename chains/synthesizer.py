from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts.templates import SYNTHESIZER_PROMPT
from utils.llm import get_llm

class SynthesizerChain:
    """Generates the final grounded answer."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.prompt = ChatPromptTemplate.from_template(SYNTHESIZER_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()
        
    def invoke(self, question: str, context: str) -> str:
        return self.chain.invoke({"question": question, "context": context})
