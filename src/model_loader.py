"""
Model loading utilities for embeddings and Gemini models.
"""

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import ollama

from src.utils import get_shared_logger

logger = get_shared_logger(__name__)


def load_embedder(fast_embed_name: str, embedding_cache_dir: str, device: str):
    """
    Load the embedding model with caching.

    Args:
        fast_embed_name (str): Name of the embedding model
        embedding_cache_dir (str): Directory to cache embeddings
        device (str): Device to load the embedder on

    Returns:
        CacheBackedEmbeddings: Cached embedding model
    """
    cache_dir = embedding_cache_dir

    logger.info(f"Loading embedding model: {fast_embed_name} on device: {device}")

    # Initialize base embeddings
    base_embedder = HuggingFaceEmbeddings(
        model_name=fast_embed_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create a cached embedder
    store = LocalFileStore(f"{cache_dir}/embeddings_cache")
    embedder = CacheBackedEmbeddings.from_bytes_store(
        base_embedder, store, namespace=fast_embed_name
    )

    logger.info(f"Cached embedder ({fast_embed_name}) loaded successfully")
    return embedder


def load_data_and_index(pc_index: str):
    """
    Load the Pinecone index.

    Args:
        pc_index (str): Name of the Pinecone index

    Returns:
        Pinecone index
    """
    # Load the Pinecone index
    logger.info(f"Connecting to Pinecone index: {pc_index}")
    pinecone = Pinecone()
    index = pinecone.Index(pc_index)
    logger.info(f"Pinecone index ({pc_index}) connected successfully")

    return index


def setup_client(model_name: str):
    """
    Load Ollama client.

    Args:
        model_name (str): Name of the Ollama model to load
    Returns:
        Ollama client
    """

    logger.info(f"Setting up Ollama client with model: {model_name}")
    client = ollama.Client(model=model_name)

    return client


def load_all_components(config: dict, device: str):
    """
    Load all necessary components for the RAG system.

    Args:
        config (dict): Configuration dictionary
        device (str): Device to load models on

    Returns:
        tuple: (index, client, embedder)
    """
    logger.info("Starting component loading process...")

    try:

        # Load data and index
        index = load_data_and_index(
            pc_index=config["pc_index"]
        )

        # Create main LLM (Ollama client)
        client = setup_client(model_name=config["model_name"])

        # Load embedder
        embedder = load_embedder(
            fast_embed_name=config["fast_embed_name"],
            embedding_cache_dir=config["embedding_cache_dir"],
            device=device
        )

        logger.info("All components loaded successfully!")
        return index, client, embedder
    
    except Exception as e:
        logger.error("Error loading components: %s", e)
        raise RuntimeError("Failed to load all components.") from e