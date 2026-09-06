from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from prompts.templates import GRADER_PROMPT
from utils.models import GraderDecision
from utils.llm import get_llm

class GraderChain:
    """Grades whether retrieved context is sufficient."""
    
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=GraderDecision)
        
        template = GRADER_PROMPT + "\n\n{format_instructions}"
        self.prompt = ChatPromptTemplate.from_template(template)
        
        self.chain = self.prompt | self.llm | self.parser
        
    def invoke(self, question: str, context: str) -> GraderDecision:
        return self.chain.invoke({
            "question": question, 
            "context": context,
            "format_instructions": self.parser.get_format_instructions()
        })
