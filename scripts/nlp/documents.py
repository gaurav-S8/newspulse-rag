# Import Libraries
import pandas as pd
from typing import Optional, Dict

def prepare_documents(
    row: pd.Series
) -> Optional[Dict]:
    """
    Convert a single preprocessed article row into a RAG-ready document object.

    This function performs lightweight validation and transforms a pandas DataFrame
    row into a standardized document dictionary containing:
    - page_content: text used for embedding (article summary)
    - metadata: structured fields used for filtering, attribution, and traceability

    Rows with missing or empty summary text are skipped by returning None.

    Parameters
    ----------
    row: pandas.Series
        A single row from the processed articles DataFrame.

    Returns
    -------
    dict or None
        A document dictionary with `page_content` and `metadata` keys,
        or None if the row fails validation.
    """
    
    summary = row.get("summary")

    # Local validation: text must exist
    if not isinstance(summary, str):
        return None

    summary = summary.strip()
    if not summary:
        return None

    return {
        "page_content": summary,
        "metadata": {
            "page_content": summary,
            "article_id": row.get("article_id"),
            "title": row.get("title"),
            "source_name": row.get("source_name"),
            "published_date": row.get("published_date"),
            "sentiment_label": row.get("sentiment_label"),
            "severity": row.get("severity"),
            "persons": row.get("persons") or [],
            "organizations": row.get("organizations") or [],
            "locations": row.get("locations") or [],
            "url": row.get("url"),
        }
    }