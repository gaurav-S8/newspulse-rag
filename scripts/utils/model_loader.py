# Import Libraries
from typing import Dict

# Import DL Libraries
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

def load_models(
    config: Dict
) -> Dict:
    """
    Load and initialize all ML models and pipelines required for the NLP pipeline.

    Parameters
    ----------
    config: Dict
        Parsed configuration dictionary containing model names under config["models"].

    Returns
    -------
    Dict
        Dictionary containing the following keys:
        - 'device': int — 0 for GPU, -1 for CPU
        - 'summarizer_pipeline': Pipeline — HuggingFace summarization pipeline
        - 'summarizer_tokenizer': PreTrainedTokenizer — tokenizer for summarization model
        - 'sentiment_pipeline': Pipeline — HuggingFace sentiment analysis pipeline
        - 'ner_pipeline': Pipeline — HuggingFace NER pipeline
        - 'keybert_model': KeyBERT — keyword extraction model built on top of the embedding model
        - 'embedding_model': SentenceTransformer — model for generating embeddings
    
    """
    
    # Device
    device = 0 if torch.cuda.is_available() else -1

    # Load all models
    # Model used for Summarization
    summarizer = config["models"]["summarization"]
    summarizer_tokenizer = AutoTokenizer.from_pretrained(config["models"]["summarization"])
    
    # Model used for Sentiment Analysis
    sentiment_analyzer = config["models"]["sentiment"]
    
    # Model used for Named Entity Recognition
    ner_tokenizer = AutoTokenizer.from_pretrained(config["models"]["ner"])
    ner_model = AutoModelForTokenClassification.from_pretrained(config["models"]["ner"])
    
    # Model used for creating Embeddings
    embedding_model = SentenceTransformer(
        config["models"]["embeddings"],
        device = device
    )

    # Create pipelines
    summarizer_pipeline = pipeline(
        "summarization",
        model = summarizer,
        tokenizer = summarizer,
        device = device
    )
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model = sentiment_analyzer,
        tokenizer = sentiment_analyzer,
        device = device
    )

    ner_pipeline = pipeline(
        "ner",
        model = ner_model,
        tokenizer = ner_tokenizer,
        aggregation_strategy = "first",
        device = device
    )

    keybert_model = KeyBERT(model = config["models"]["embeddings"])

    return {
        'device': device,
        'ner_pipeline': ner_pipeline,
        'summarizer_tokenizer': summarizer_tokenizer,
        'summarizer_pipeline': summarizer_pipeline,
        'sentiment_pipeline': sentiment_pipeline,
        'keybert_model': keybert_model,
        'embedding_model': embedding_model
    }