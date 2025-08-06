import os
from pathlib import Path
import shutil

from chonkie import AutoEmbeddings, CodeChunker, RecursiveChunker
from dotenv import load_dotenv

from rerankers import Reranker

from app.embedding.azure_openai import AzureOpenAIEmbeddings


# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

env_file = BASE_DIR / ".env"
load_dotenv(env_file)


def is_ffmpeg_installed():
    """
    Check if ffmpeg is installed on the current system.
    
    Returns:
        bool: True if ffmpeg is installed, False otherwise.
    """
    return shutil.which("ffmpeg") is not None



class Config:
    # Check if ffmpeg is installed
    if not is_ffmpeg_installed():
        import static_ffmpeg
        # ffmpeg installed on first call to add_paths(), threadsafe.
        static_ffmpeg.add_paths()
        # check if ffmpeg is installed again
        if not is_ffmpeg_installed():
            raise ValueError("FFmpeg is not installed on the system. Please install it to use the Surfsense Podcaster.")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    NEXT_FRONTEND_URL = os.getenv("NEXT_FRONTEND_URL")
    
    
    # LLM instances are now managed per-user through the LLMConfig system
    # Legacy environment variables removed in favor of user-specific configurations

    # Chonkie Configuration | Edit this to your needs
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_model_instance = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model=EMBEDDING_MODEL,
        deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT","text-embedding-3-small"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        dimension= int(os.getenv("EMBEDDING_DIMENSION", 1024)))
    chunker_instance = RecursiveChunker(
        chunk_size=getattr(embedding_model_instance, 'max_seq_length', 512)
    )
    code_chunker_instance = CodeChunker(
        chunk_size=getattr(embedding_model_instance, 'max_seq_length', 512)
    )
    
    # Reranker's Configuration | Pinecode, Cohere etc. Read more at https://github.com/AnswerDotAI/rerankers?tab=readme-ov-file#usage
    
    # Check for Azure reranker configuration first
    AZURE_RERANKERS_ENDPOINT = os.getenv("AZURE_RERANKERS_ENDPOINT")
    AZURE_RERANKERS_ENDPOINT_API_KEY = os.getenv("AZURE_RERANKERS_ENDPOINT_API_KEY")
    
    if AZURE_RERANKERS_ENDPOINT and AZURE_RERANKERS_ENDPOINT_API_KEY:
        # Use Azure reranker if both endpoint and API key are configured
        from app.reranker.azureReranker import AzureReranker
        reranker_instance = AzureReranker(
            model_name="azure-cohere-rerank",
            endpoint=AZURE_RERANKERS_ENDPOINT,
            api_key=AZURE_RERANKERS_ENDPOINT_API_KEY
        )
    else:
        # Fallback to standard rerankers
        RERANKERS_MODEL_NAME = os.getenv("RERANKERS_MODEL_NAME")
        RERANKERS_MODEL_TYPE = os.getenv("RERANKERS_MODEL_TYPE")
        
        if RERANKERS_MODEL_NAME:
            reranker_instance = Reranker(
                model_name=RERANKERS_MODEL_NAME,
                model_type=RERANKERS_MODEL_TYPE,
            )
        else:
            # No reranker configured
            reranker_instance = None


    
    # OAuth JWT
    SECRET_KEY = os.getenv("SECRET_KEY")
    
    # ETL Service
    ETL_SERVICE = os.getenv("ETL_SERVICE")
    
    if ETL_SERVICE == "UNSTRUCTURED":
        # Unstructured API Key
        UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
        
    elif ETL_SERVICE == "LLAMACLOUD":
        # LlamaCloud API Key
        LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
        
        
    # Firecrawl API Key
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", None) 
    
    # Litellm TTS Configuration
    TTS_SERVICE = os.getenv("TTS_SERVICE")
    TTS_SERVICE_API_BASE = os.getenv("TTS_SERVICE_API_BASE")
    TTS_SERVICE_API_KEY = os.getenv("TTS_SERVICE_API_KEY")
    
    # Litellm STT Configuration
    STT_SERVICE = os.getenv("STT_SERVICE")
    STT_SERVICE_API_BASE = os.getenv("STT_SERVICE_API_BASE")
    STT_SERVICE_API_KEY = os.getenv("STT_SERVICE_API_KEY")
    
    
    # Validation Checks
    # Check embedding dimension
    if hasattr(embedding_model_instance, 'dimension') and embedding_model_instance.dimension > 2000:
        raise ValueError(
            f"Embedding dimension for Model: {EMBEDDING_MODEL} "
            f"has {embedding_model_instance.dimension} dimensions, which "
            f"exceeds the maximum of 2000 allowed by PGVector."
        )


    @classmethod
    def get_settings(cls):
        """Get all settings as a dictionary."""
        return {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }


# Create a config instance
config = Config()
