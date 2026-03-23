# Import Libraries
import hashlib
import unicodedata
import pandas as pd
from typing import List, Dict

def normalize_punctuation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Unicode punctuation in article text to ASCII equivalents.

    Replaces curly quotes, apostrophes, dashes, and other Unicode punctuation
    with their standard ASCII forms. Should be applied before any NLP processing
    to ensure consistent text across summarization, sentiment, and NER pipelines.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame containing a 'news_content' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized 'news_content' column.
    """
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u02BC': "'",
        '\u201C': '"', '\u201D': '"',
        '\u2013': '-', '\u2014': '-',
        '\u2026': '...',
    }
    for unicode_char, ascii_char in replacements.items():
        df['news_content'] = df['news_content'].str.replace(unicode_char, ascii_char, regex=False)
    return df

def drop_post_nlp_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove intermediate NLP columns that are not needed for downstream tasks.

    Drops columns generated during the NLP pipeline that are either
    redundant or not required for retrieval and display.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame after full NLP pipeline has been applied.

    Returns
    -------
    pd.DataFrame
        DataFrame with intermediate columns removed.
    """
    
    columns_to_drop = [
        col for col in df.columns
        if col.lower() in {'news_content', 'summary_token_count', 'summary_word_count', 'sentiment_score'}
    ]
    return df.drop(columns = columns_to_drop)

def preprocess_news_articles(
    articles: List[Dict],
    min_text_length: int
) -> pd.DataFrame:
    """
    Preprocess raw NewsAPI articles into a clean, structured DataFrame.

    This function performs source parsing, text consolidation, date normalization,
    and filtering of invalid or short articles to prepare the data for downstream NLP tasks.

    Parameters
    ----------
    articles: list of dict
        List of raw article objects returned by NewsAPI.
    min_text_length: int
        Minimum character length required for article text.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame containing processed article content and metadata, ready for NLP processing.
    """

    if not isinstance(articles, list):
        raise TypeError(f"'articles' must be a list, got {type(articles)}")
    
    df = pd.DataFrame(articles)
    # Standardize columns
    df.columns = [
        col.lower() for col in df.columns
    ]
    
    required_columns = ['source', 'title', 'description', 'publishedat']
    missing = [col for col in required_columns if col not in df.columns]
    if(missing):
        raise ValueError(f"Missing required columns: {missing}")

    # Split 'SOURCE' column in two different columns - 'SOURCE_ID', 'SOURCE_NAME'
    df['source_id'] = df['source'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
    df['source_name'] = df['source'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)

    # Combine Title and Description column to create a new column 'NEWS_CONTENT'
    df['news_content'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['news_content'] = df['news_content'].str.strip()

    # Remove very small articles
    df['news_content'] = df['news_content'].where(df['news_content'].str.len() >= min_text_length, None)
    
    # Remove rows with "None" as "NEWS_CONTENT"
    df = df[df['news_content'].notna()]
    if df.empty:
        return df

    # Standardize texts
    df = normalize_punctuation(df)

    # Get 'PUBLISHED_DATE' from 'PublishedAt' column
    df['published_date'] = pd.to_datetime(
        df['publishedat'], errors = 'coerce'
    ).dt.date

    # Create unique Article ID
    df['article_id'] = df[['title', 'source_name', 'published_date']].apply(
        lambda row: hashlib.md5(
            (str(row['title']) + str(row['source_name']) + str(row['published_date'])).encode()
        ).hexdigest(),
        axis = 1
    )
    
    # Remove useless columns
    # --- can be kept if needed in the future
    columns_to_drop = [
        col for col in df.columns
        if col.lower() in {'source', 'author', 'description', 'urltoimage', 'content', 'source_id', 'publishedat'}
    ]
    df = df.drop(columns = columns_to_drop)
    return df