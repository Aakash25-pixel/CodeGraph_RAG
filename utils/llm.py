from langchain_core.language_models.chat_models import BaseChatModel
from utils.config import LLM_MODEL_NAME, USE_HUGGINGFACE, HF_TOKEN

def get_llm(temperature: float = 0.0, max_tokens: int = 1200, timeout: int = 120) -> BaseChatModel:
    """Returns the configured LLM instance."""
    if USE_HUGGINGFACE and HF_TOKEN:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        
        # Default to a highly capable free tier chat model if not specified
        model_id = LLM_MODEL_NAME if "/" in LLM_MODEL_NAME else "Qwen/Qwen2.5-Coder-32B-Instruct"
        
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            temperature=temperature if temperature > 0 else 0.1,  # HF often requires temp > 0
            max_new_tokens=max_tokens,
            huggingfacehub_api_token=HF_TOKEN,
            timeout=timeout
        )
        return ChatHuggingFace(llm=llm)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LLM_MODEL_NAME, temperature=temperature)
