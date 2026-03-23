# Import Libraries
import pandas as pd
from transformers import Pipeline, PreTrainedTokenizerBase

def summarize_articles(
    df: pd.DataFrame,
    min_text_length: int,
    max_text_length: int,
    summarizer_pipeline: Pipeline,
    tokenizer: PreTrainedTokenizerBase,
    summarize: bool = True
) -> pd.DataFrame:
    """
    Generate abstractive summaries for news articles using batch inference.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame containing a `news_content` column with article text.
    min_text_length: int
        Minimum length of the generated summary (in tokens).
    max_text_length: int
        Maximum length of the generated summary (in tokens).
    summarizer_pipeline: Pipeline
        Preloaded Hugging Face summarization pipeline.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer corresponding to the summarization model, used for token counting.
    summarize: bool, optional
        Whether to run summarization on articles. Defaults to True.
        If False, raw article content is used directly.
    
    Returns
    -------
    pandas.DataFrame
        The input DataFrame with three additional columns:
        - `summary`: generated article summary or raw content if summarize=False
        - `summary_token_count`: number of tokens in the summary
        - `summary_word_count`: number of words in the summary
    """
    if 'news_content' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'news_content' column.")

    texts = df['news_content'].tolist()
    texts = [text[:4000] for text in texts]

    if summarize:
        result = summarizer_pipeline(
            texts,
            batch_size = 16,
            max_length = max_text_length,
            min_length = min_text_length,
            do_sample = False
        )
        summaries = [r['summary_text'] for r in result]
    else:
        summaries = texts

    token_counts = [len(tokenizer.tokenize(s)) for s in summaries]
    word_counts = [len(s.split()) for s in summaries]

    df['summary'] = summaries
    df['summary_token_count'] = token_counts
    df['summary_word_count'] = word_counts
    return df