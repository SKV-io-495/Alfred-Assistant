"""LLM factory module for alfred backend."""

from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from .config import config

def get_llm() -> ChatGroq:
    """Get configured GROQ LLM instance."""
    return ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.GROQ_MODEL_NAME,
        temperature=0.1,
        max_tokens=2048,
        timeout=60,
        max_retries=2
    )

def get_embeddings():
    # Limit threads to 1 to prevent OOM kills on Render's 512MB free tier
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=1)