# Import Libraries
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

def save_vector_store(
    faiss_index: faiss.Index,
    metadata_store: List[Dict], 
    save_dir: str,
    index_name: str,
    metadata_name: str
) -> None:
    """
    Persist FAISS index and aligned metadata store to disk.

    Parameters
    ----------
    faiss_index: faiss.Index
        Trained FAISS vector index.
    metadata_store: list of dict
        Metadata aligned 1:1 with FAISS index positions.
    save_dir: str or Path, optional
        Directory where vector store artifacts will be saved.
    index_name: str, optional
        Filename for FAISS index.
    metadata_name: str, optional
        Filename for metadata store.

    Returns
    -------
    None
    """
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents = True, exist_ok = True)

    # Save FAISS index
    faiss.write_index(faiss_index, str(save_dir / index_name))

    # Save metadata store
    with open(save_dir / metadata_name, "wb") as f:
        pickle.dump(metadata_store, f)

def load_vector_store(
    vector_store_path: str
) -> Tuple[Optional[faiss.Index], Optional[List[Dict]]]:
    """
    Load a persisted FAISS index and metadata store from disk.

    Parameters
    ----------
    vector_store_path: str
        Path to the directory containing the FAISS index and metadata store.

    Returns
    -------
    Tuple[faiss.Index, List[Dict]] or Tuple[None, None]
        - FAISS index loaded from disk
        - Metadata store aligned with FAISS index positions
        Returns (None, None) if the vector store does not exist yet.
    """
    
    index_path = Path(vector_store_path) / "faiss.index"
    metadata_path = Path(vector_store_path) / "metadata.pkl"

    if not index_path.exists() or not metadata_path.exists():
        return None, None
    
    faiss_index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata_store = pickle.load(f)
    
    return faiss_index, metadata_store

def update_vector_store(
    new_faiss_index: faiss.Index,
    new_metadata: List[Dict],
    save_dir: str,
    index_name: str = "faiss.index",
    metadata_name: str = "metadata.pkl"
) -> None:
    """
    Add new embeddings to an existing FAISS index and persist to disk.
    Deduplicates articles by article_id before adding.
    If no existing index is found, saves the new one directly.

    Parameters
    ----------
    new_faiss_index: faiss.Index
        Newly built FAISS index from the latest batch of articles.
    new_metadata: List[Dict]
        Metadata aligned with the new FAISS index.
    save_dir: str
        Directory where vector store artifacts are saved.
    index_name: str, optional
        Filename for FAISS index. Defaults to "faiss.index".
    metadata_name: str, optional
        Filename for metadata store. Defaults to "metadata.pkl".

    Returns
    -------
    None
    """
    existing_index, existing_metadata = load_vector_store(save_dir)

    if existing_index is not None:
        # Deduplicate by article_id
        existing_ids = {m.get("article_id") for m in existing_metadata}
        
        new_indices_to_add = [
            i for i, m in enumerate(new_metadata)
            if m.get("article_id") not in existing_ids
        ]

        if not new_indices_to_add:
            # Nothing new to add
            return

        # Extract only the new unique vectors
        all_new_vectors = np.array(
            [
                new_faiss_index.reconstruct(i)
                for i in range(new_faiss_index.ntotal)
            ], dtype = "float32"
        )
        unique_vectors = all_new_vectors[new_indices_to_add]
        unique_metadata = [new_metadata[i] for i in new_indices_to_add]

        # Add to existing index
        existing_index.add(unique_vectors)
        existing_metadata.extend(unique_metadata)

        save_vector_store(existing_index, existing_metadata, save_dir, index_name, metadata_name)
    else:
        save_vector_store(new_faiss_index, new_metadata, save_dir, index_name, metadata_name)