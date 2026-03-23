# Import Libraries
import os
import json
import pandas as pd
from typing import List

def save_dataframe(
    df: pd.DataFrame,
    output_path: str,
    file_format: str = "csv"
) -> None:
    """
    Save a processed pandas DataFrame to a file.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the processed and enriched news data.
    output_path: str
        File path where the file will be saved.
    file_format: str, optional
        Output format — 'csv', 'json', or 'parquet'. Defaults to 'csv'.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an unsupported format is provided.
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok = True)

    if(file_format == "csv"):
        df.to_csv(output_path, index = False)
    elif(file_format == "json"):
        df.to_json(output_path, orient = "records", lines = True)
    elif(file_format == "parquet"):
        df.to_parquet(output_path, index = False)
    else:
        raise ValueError(f"Unsupported format '{file_format}'. Choose from 'csv', 'json', or 'parquet'.")

def save_fetched_articles(
    articles: List,
    file_path: str
) -> None:
    """
    Save fetched NewsAPI articles to a JSON file.

    Parameters
    ----------
    articles: list
        List of article objects returned by NewsAPI.
    file_path: str
        Full file path (with .json extension) where the data will be saved.

    Returns
    -------
    None
    """
    
    os.makedirs(os.path.dirname(file_path), exist_ok = True)
    with open(file_path, 'w', encoding = 'utf-8') as f:
        json.dump(articles, f, ensure_ascii = False, indent = 2)