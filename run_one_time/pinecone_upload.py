"""
Embedding of text data and upload to Pinecone index.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from src.utils import get_shared_logger, load_config


logger = get_shared_logger(__name__)

def _strip_possessive(value: str) -> str:
    return re.sub(r"(?:'s|’s)$", "", value.strip())


def _tokenize_name(normalized_value: str) -> list[str]:
    tokens = [token for token in re.split(r"\W+", normalized_value.lower()) if token]
    return tokens


def embed_text(
    data_dir_path,
    texts: Sequence[str],
    batch_size,
    save_every,
    save_checkpoints,
    fast_embed_name,
    embeddings_file_name,
):
    """
    Embeds text data using a specified text embedding model and saves the embeddings to a file.
    """
    print("EMBEDDING START!")
    os.makedirs(data_dir_path, exist_ok=True)
    # Check for CUDA availability
    providers = ort.get_available_providers()
    print("Available providers:", providers)
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "CUDAExecutionProvider not available. Install onnxruntime-gpu and ensure CUDA drivers are configured."
        )

    # Create the text embedding model
    embedding_model = TextEmbedding(
        model_name=fast_embed_name,
        batch_size=batch_size,  # This controls how many texts are embedded at once
        providers=["CUDAExecutionProvider"],
    )

    # Embed the texts in batches and save checkpoints
    all_embeddings = []
    texts_list = list(texts)

    for i in tqdm(range(0, len(texts_list), save_every)):
        checkpoint_texts = texts_list[i : i + save_every]
        # The embedding model will internally process these in batches of batch_size
        batch_embeddings = list(embedding_model.embed(checkpoint_texts))

        # Normalize embeddings here
        normalized_batch = []
        for vec in batch_embeddings:
            norm = np.linalg.norm(vec)
            if norm == 0:
                normalized_batch.append(np.array(vec, dtype=np.float32))
            else:
                normalized_batch.append(np.array(vec, dtype=np.float32) / norm)

        all_embeddings.extend(normalized_batch)

        if save_checkpoints and i > 0:
            # Save checkpoint after each save_every texts
            np.save(
                os.path.join(data_dir_path, f"embeddings_checkpoint_{i}.npy"),
                np.array(all_embeddings, dtype=np.float32),
            )

    print("Finished embedding text")

    # Save final embeddings
    final_embeddings = np.array(all_embeddings, dtype=np.float32)
    np.save(os.path.join(data_dir_path, embeddings_file_name), final_embeddings)

    print(f"Saved final embeddings to {data_dir_path}")

    return final_embeddings


def _check_and_create_index(
    pc: Pinecone,
    pc_index: str,
    embeddings: np.ndarray,
    distance_metric: str,
    pc_cloud: str,
    pc_region: str,
) -> None:
    """
    Check if a Pinecone index exists, delete if it does, and create a new one.

    Args:
        pc (Pinecone): Pinecone client instance
        pc_index (str): Index name
        embeddings (np.ndarray): Embeddings array to determine dimension
        distance_metric (str): Metric for the index
        pc_cloud (str): Cloud provider
        pc_region (str): Cloud region
    """
    logger.info(f"Checking for index: {pc_index}")
    existing_indexes = pc.list_indexes().names()
    if pc_index in existing_indexes:
        logger.info(f"Deleting index: '{pc_index}'")
        pc.delete_index(pc_index)
    logger.info(f"Creating index: '{pc_index}'")
    pc.create_index(
        name=pc_index,
        dimension=embeddings.shape[1],
        metric=distance_metric,
        spec=ServerlessSpec(cloud=pc_cloud, region=pc_region),
    )


def _prepare_vectors(
    embeddings: np.ndarray, metadata_records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Prepare vectors for upsert to Pinecone.

    Args:
        embeddings (np.ndarray): Embeddings array
        metadata_records (Sequence[dict[str, Any]]): Corresponding metadata records

    Returns:
        list[dict[str, Any]]: List of vectors with id, values, and metadata
    """
    if len(embeddings) != len(metadata_records):
        raise ValueError("Embeddings count does not match metadata count.")

    vectors: list[dict[str, Any]] = []
    for idx in tqdm(range(len(embeddings)), desc="Preparing vectors"):
        embedding = embeddings[idx]
        metadata = metadata_records[idx]
        vector_id = str(metadata.get("id"))
        if vector_id == "None":
            raise ValueError("Message metadata missing 'id' field; cannot build vector id.")
        metadata_enriched = dict(metadata)
        message_text = metadata_enriched.get("message", "")
        metadata_enriched.setdefault("text", message_text)

        user_name = metadata_enriched.get("user_name")
        if isinstance(user_name, str) and user_name.strip():
            normalized = _strip_possessive(user_name).strip()
            normalized_lower = normalized.lower()
            tokens = _tokenize_name(normalized_lower)
            metadata_enriched["user_name_normalized"] = normalized_lower
            if tokens:
                metadata_enriched["user_name_tokens"] = tokens
        vectors.append(
            {
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": metadata_enriched,
            }
        )
    return vectors


def pinecone_upload(
    pc_index: str,
    embeddings: np.ndarray,
    metadata_records: Sequence[dict[str, Any]],
    batch_size: int,
    distance_metric: str,
    pc_cloud: str,
    pc_region: str,
) -> None:
    """
    Upload embeddings to Pinecone index in batches.

    Args:
        pc_index (str): Index name
        embeddings (np.ndarray): Embeddings array
        batch_size (int): Batch size for upsert
        distance_metric (str): Metric for the index
        pc_cloud (str): Cloud provider
        pc_region (str): Cloud region
    """
    logger.info("PINECONE UPLOAD START!")
    pc = Pinecone()
    _check_and_create_index(
        pc, pc_index, embeddings, distance_metric, pc_cloud, pc_region
    )
    index = pc.Index(pc_index)
    vectors = _prepare_vectors(embeddings, metadata_records)
    for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting batches"):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
    logger.info("Index upserted successfully!")


def _load_messages(messages_path: Path) -> list[dict[str, Any]]:
    if not messages_path.exists():
        raise FileNotFoundError(f"Messages file not found at {messages_path}")
    with messages_path.open("r", encoding="utf-8") as messages_file:
        messages: list[dict[str, Any]] = json.load(messages_file)
    if not messages:
        raise ValueError("No messages found to embed.")
    return messages


def main():
    load_dotenv()

    config = load_config()

    data_dir = Path(config["data_dir"]).resolve()
    messages_path = data_dir / "all_messages.json"
    messages = _load_messages(messages_path)

    texts = [msg["message"] for msg in messages if "message" in msg]
    if len(texts) != len(messages):
        raise ValueError("Some messages are missing the 'message' field.")

    save_checkpoints_raw = config["embedding_save_checkpoints"]
    if isinstance(save_checkpoints_raw, str):
        save_checkpoints = save_checkpoints_raw.strip().lower() in {"1", "true", "yes"}
    else:
        save_checkpoints = bool(save_checkpoints_raw)

    embeddings = embed_text(
        data_dir_path=config["data_dir"],
        texts=texts,
        batch_size=int(config["embedding_batch_size"]),
        save_every=int(config["embedding_save_every"]),
        save_checkpoints=save_checkpoints,
        fast_embed_name=config["fast_embed_name"],
        embeddings_file_name=config["embeddings_file_name"],
    )

    if embeddings.size == 0:
        raise ValueError("Embedding array is empty; aborting upload.")

    if not os.getenv("PINECONE_API_KEY"):
        raise EnvironmentError("PINECONE_API_KEY is not set in the environment.")

    pinecone_upload(
        pc_index=config["pc_index"],
        embeddings=embeddings,
        metadata_records=messages,
        batch_size=int(config["pinecone_batch_size"]),
        distance_metric=config["distance_metric"],
        pc_cloud=config["pc_cloud"],
        pc_region=config["pc_region"],
    )


if __name__ == "__main__":
    main()
