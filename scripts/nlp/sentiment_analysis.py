# Import Libraries
import pandas as pd
from transformers import Pipeline

SEVERITY_MAP = {'Negative': 'High', 'Neutral': 'Medium'}

def apply_sentiment_analysis(
    df: pd.DataFrame,
    sentiment_pipeline: Pipeline
) -> pd.DataFrame:
    """
    Apply sentiment analysis and severity classification to article summaries using batch inference.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame containing a `summary` column with article summaries.
    sentiment_pipeline: Pipeline
        Preloaded Hugging Face sentiment analysis pipeline.
    
    Returns
    -------
    pandas.DataFrame
        The input DataFrame with three additional columns:
        - `sentiment_label`: predicted sentiment label (Positive, Negative, Neutral)
        - `sentiment_score`: confidence score for the prediction
        - `severity`: qualitative severity level derived from sentiment (High, Medium, Low)
    """
    
    if 'summary' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'summary' column.")

    summaries = df['summary'].tolist()
    result = sentiment_pipeline(
        summaries,
        batch_size = 8
    )
    
    sentiment_labels, sentiment_scores, severities = [], [], []
    for r in result:
        label, score = r['label'].capitalize(), r['score']
        sentiment_labels.append(label)
        sentiment_scores.append(score)
        severities.append(SEVERITY_MAP.get(label, 'Low'))
    
    df['sentiment_label'] = sentiment_labels
    df['sentiment_score'] = sentiment_scores
    df['severity'] = severities
    return df