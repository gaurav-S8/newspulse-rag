# Import Libraries
import pandas as pd
from typing import Dict
from transformers import Pipeline, PreTrainedTokenizerBase

# Import Custom Modules
from scripts.ingest_articles.fetch_news import (
    get_everything, # get_top_headlines, get_top_headline_sources
)
from scripts.nlp.summarization import summarize_articles
from scripts.nlp.ner import apply_named_entity_recognition
from scripts.nlp.sentiment_analysis import apply_sentiment_analysis
from scripts.preprocessing.article_preprocessing import preprocess_news_articles

def run_news_pipeline(
    query_params: Dict,
    api_key: str,
    log_file_path: str,
    raw_data_path: str,
    sentiment_pipeline: Pipeline,
    summarizer_tokenizer: PreTrainedTokenizerBase,
    summarization_pipeline: Pipeline,
    ner_pipeline: Pipeline,
    default_min_text_length: int,
    default_max_text_length: int,
    summarize: bool = True
) -> pd.DataFrame:
    """
    Execute the complete end-to-end NLP pipeline on news articles.
    
    This function orchestrates the full workflow starting from data ingestion via NewsAPI,
    followed by preprocessing, summarization, sentiment analysis, and NER.
    It returns a single enriched DataFrame containing all derived information.
    
    Parameters
    ----------
    query_params: dict
        Dictionary containing parameters required to query the NewsAPI endpoint,
        including endpoint type, search query, date range, language, and filters.
    api_key: str
        NewsAPI authentication key.
    log_file_path: str
        Path to the log file for request logging.
    raw_data_path: str
        Path to save raw article data.
    sentiment_pipeline: Pipeline
        Preloaded Hugging Face sentiment analysis pipeline.
    summarizer_tokenizer: PreTrainedTokenizerBase
        Tokenizer corresponding to the summarization model, used for token counting.
    summarization_pipeline: Pipeline
        Preloaded Hugging Face summarization pipeline.
    ner_pipeline: Pipeline
        Preloaded Hugging Face NER pipeline.
    default_min_text_length: int
        Minimum character/token length for text filtering and summarization.
    default_max_text_length: int
        Maximum token length for generated summaries.
    summarize: bool, optional
        Whether to run summarization on articles. Defaults to True.
        If False, raw article content is used directly.
    
    Returns
    -------
    pandas.DataFrame
        Final enriched DataFrame containing:
        - cleaned article text
        - generated summaries
        - sentiment labels, scores, and severity
        - extracted named entities
    """

    if(not isinstance(query_params, dict)):
        raise TypeError("query_params must be a dictionary")

    endpoint = query_params.get('end_point')
    if endpoint not in {"https://newsapi.org/v2/everything", "https://newsapi.org/v2/top-headlines"}:
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    if endpoint == "https://newsapi.org/v2/everything":
        data = get_everything(
            endpoint,
            api_key,
            log_file_path,
            raw_data_path,
            q = query_params.get('q'),
            search_in = query_params.get('search_in'),
            sources = query_params.get('source'),
            domains = query_params.get('domain'),
            exclude_domains = query_params.get('excluded_domains'),
            from_date = query_params.get('from_date'),
            to_date = query_params.get('to_date'),
            language = query_params.get('language'),
            sort_by = query_params.get('sort_by'),
            page_size = 32,
            page = 1
        )
    elif endpoint == "https://newsapi.org/v2/top-headlines":
        raise NotImplementedError("top_headlines endpoint not implemented yet")

    if not data or data.get('status') != 'ok':
        raise RuntimeError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
    
    if len(data['articles']) == 0:
        raise RuntimeError("NewsAPI returned 0 articles. Try adjusting the date range or query.")
    
    df = preprocess_news_articles(data['articles'], default_min_text_length)
    if df.empty:
        raise RuntimeError("No usable news articles found. Try adjusting your date range or query.")
    
    df = summarize_articles(df, default_min_text_length, default_max_text_length, summarization_pipeline, summarizer_tokenizer, summarize)
    df = apply_sentiment_analysis(df, sentiment_pipeline)
    df = apply_named_entity_recognition(df, ner_pipeline)
    return df