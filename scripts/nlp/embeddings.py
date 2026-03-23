# Import Libraries
import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

def generate_embeddings(
    embedding_model: SentenceTransformer,
    texts: List[str]
) -> np.ndarray:
    """
    Generate dense vector embeddings for a list of text inputs.

    Parameters
    ----------
    embedding_model: SentenceTransformer
        Preloaded SentenceTransformer model used to generate dense vector embeddings.
    texts: List[str]
        List of text strings to embed (e.g., document summaries).

    Returns
    -------
    np.ndarray
        Array of shape (n_texts, embedding_dim) containing float32 embeddings.
    """

    if(isinstance(texts, str)):
        texts = [texts]

    if not isinstance(texts, list) or not texts:
        raise ValueError("`texts` must be a non-empty string or list of strings")
    
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy = True,
        show_progress_bar = True
    )
    return embeddings.astype("float32")

def build_embeddings(
    embedding_model: SentenceTransformer,
    documents: List[Dict]
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Build a FAISS vector index and aligned metadata store from chunked documents.

    Parameters
    ----------
    embedding_model: SentenceTransformer
        Model used for creating embeddings
    documents: List[Dict]
        List of chunked document objects containing:
        - "page_content": str
        - "metadata": dict

    Returns
    -------
    Tuple[faiss.Index, List[Dict]]
        - FAISS index containing document embeddings
        - Metadata store aligned with FAISS index positions
    """
    
    if not documents:
        raise ValueError("documents list is empty — nothing to embed.")
    
    texts = [doc["page_content"] for doc in documents]
    embeddings = generate_embeddings(embedding_model, texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    metadata_store = [doc["metadata"] for doc in documents]
    return index, metadata_store