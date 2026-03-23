# Import Libraries
import faiss
import numpy as np
from typing import List, Dict, Tuple

def similarity_search(
    faiss_index: faiss.Index,
    query_vectors: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform similarity search in a FAISS index using precomputed query embeddings.

    Parameters
    ----------
    faiss_index: faiss.Index
        FAISS index containing document embeddings.
    query_vectors: np.ndarray
        Query embeddings of shape (num_queries, embedding_dim).
    k: int, optional
        Number of nearest neighbors to retrieve per query.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        distances: shape (num_queries, k)
        indices: shape (num_queries, k)
    """

    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)

    query_vectors = query_vectors.astype("float32")

    distances, indices = faiss_index.search(query_vectors, k)
    return distances, indices

def filter_by_metadata (
    indices: np.ndarray,
    metadata_store: List[Dict], 
    filter_criteria: Dict
) -> List[int]:
    """
    Filter FAISS-retrieved document indices using metadata constraints.
    
    Parameters
    ----------
    indices: np.ndarray
        FAISS result indices.
    metadata_store: List[Dict]
        Metadata store aligned with FAISS index positions.
    filter_criteria: Dict
        Dictionary of filter constraints. Supported keys:
        - 'severity': str — e.g., "High"
        - 'sentiment_label': str — e.g., "Negative"
        - 'locations': List[str]
        - 'organizations': List[str]
        - 'persons': List[str]
        - 'from_date': date
        - 'to_date': date
    
    Returns
    -------
    List[int]
        Filtered list of FAISS indices.
    """
    
    if indices.size == 0:
        return []
    
    filtered_indices = []
    
    # Flatten FAISS indices
    indices = indices.flatten().tolist()

    # All entities
    filter_entities = (
        (filter_criteria.get('locations') or []) + 
        (filter_criteria.get('persons') or []) + 
        (filter_criteria.get('organizations') or [])
    )
    filter_entities = set(filter_entities)
    
    for i in indices:
        if i == -1:
            continue
        metadata = metadata_store[i]
        if(filter_criteria.get('severity') and metadata.get('severity') != filter_criteria.get('severity')):
            continue

        if(filter_criteria.get('sentiment_label') and metadata.get('sentiment_label') != filter_criteria.get('sentiment_label')):
            continue

        metadata_all_entities = (
            (metadata.get("locations") or []) + 
            (metadata.get("persons") or []) + 
            (metadata.get("organizations") or [])
        )

        if filter_entities and not set(metadata_all_entities).intersection(filter_entities):
            continue

        published_date = metadata.get('published_date')
        if published_date:
            if(filter_criteria.get('from_date') and published_date < filter_criteria.get('from_date')):
                continue
            if(filter_criteria.get('to_date') and published_date > filter_criteria.get('to_date')):
                continue

        filtered_indices.append(i)
    return filtered_indices