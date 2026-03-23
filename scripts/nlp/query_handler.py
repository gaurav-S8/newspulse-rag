# Import Libraries
import numpy as np
from datetime import date
from keybert import KeyBERT
from transformers import Pipeline
from typing import Dict, List, Tuple

# Import Custom Modules
from scripts.nlp.context import context_forming, generate_answer
from scripts.nlp.retriever import similarity_search, filter_by_metadata
from scripts.nlp.vector_store import load_vector_store, update_vector_store
from scripts.nlp.embeddings import build_embeddings, generate_embeddings
from scripts.nlp.ner import analyze_query
from scripts.nlp.documents import prepare_documents
from scripts.pipeline.news_pipeline import run_news_pipeline
from scripts.preprocessing.article_preprocessing import drop_post_nlp_columns

def extract_query_entities(
    user_query: str,
    ner_pipeline: Pipeline,
    keybert_model: KeyBERT
) -> Tuple[Dict, List, str]:
    """
    Extract named entities from a user query and build a search keyword.
    Parameters
    ----------
    user_query: str
        The user's search query.
    ner_pipeline: Pipeline
        Preloaded HuggingFace NER pipeline.
    keybert_model: KeyBERT
        Preloaded KeyBERT model for keyword extraction.
    Returns
    -------
    Tuple[Dict, List, str]
        - query_entities: dict with 'persons', 'locations', 'organizations' keys
        - extracted_entities: flat list of all extracted entities
        - keyword: AND-joined entity string for NewsAPI, or raw query if no entities found
    """
    
    # KeyBERT keyword extraction
    keywords = keybert_model.extract_keywords(
        user_query,
        use_mmr = True, 
        diversity = 0.7,
        top_n = 3,
        stop_words = 'english'
    )
    extracted_keywords = [kw[0] for kw in keywords]
    
    query_entities = analyze_query(user_query, ner_pipeline)
    extracted_entities = (
        (query_entities['persons'] or []) +
        (query_entities['locations'] or []) +
        (query_entities['organizations'] or []) + 
        extracted_keywords
    )
    keyword = " AND ".join(extracted_entities) if extracted_entities else user_query
    return query_entities, extracted_entities, keyword
    

def handle_user_query(
    user_query: str,
    news_api_key: str,
    default_params: Dict,
    model_pipelines: Dict,
    from_date: date,
    to_date: date,
    sort_by: str,
    query_entities: Dict,
    extracted_entities: List,
    keyword: str,
    summarize: bool
) -> Tuple[str, List]:
    """
    Execute the full query handling pipeline for a user search.

    Searches the vector store for relevant articles, fetches new articles
    from NewsAPI if not enough results are found, generates an LLM response,
    and returns results for display.

    Parameters
    ----------
    user_query: str
        The user's search query.
    news_api_key: str
        NewsAPI authentication key.
    default_params: Dict
        Configuration parameters including paths and model settings.
    model_pipelines: Dict
        Dictionary of loaded model pipelines and tokenizers.
    from_date: date
        Start date for article filtering.
    to_date: date
        End date for article filtering.
    sort_by: str
        Sorting method for NewsAPI results.
    query_entities: Dict
        Named entities extracted from the query with 'persons', 'locations',
        and 'organizations' keys.
    extracted_entities: List
        Flat list of all extracted entities.
    keyword: str
        AND-joined entity string for NewsAPI, or raw query if no entities found.
    summarize: bool, optional
        Whether to run summarization on articles. Defaults to True.
        If False, raw article content is used directly.

    Returns
    -------
    Tuple[str, List]
        - response: LLM generated answer
        - article_cards_info: list of article metadata dicts for display

    Raises
    ------
    RuntimeError
        If the news pipeline fails or no articles are available.
    """

    query_metadata = {
        'severity': None,
        'sentiment_label': None,
        'persons': query_entities.get('persons'),
        'locations': query_entities.get('locations'),
        'organizations': query_entities.get('organizations'),
        'from_date': from_date,
        'to_date': to_date
    }

    news_query_params = {
        'end_point': default_params['ep_everything'],
        'q': keyword,
        'search_in': None,
        'source': None,
        'domain': None,
        'excluded_domains': None,
        'from_date': str(from_date),
        'to_date': str(to_date),
        'language': 'en',
        'sort_by': sort_by
    }

    # Create vector embedding of the user query
    user_query_vector = generate_embeddings(
        model_pipelines['embedding_model'],
        [user_query]
    )[0]

    # Load FAISS index
    faiss_index, metadata_store = load_vector_store(default_params["vector_store_path"])
    nearest_k_indices = np.array([])
    if faiss_index is not None and metadata_store is not None:
        _, nearest_k_indices = similarity_search(
            faiss_index,
            user_query_vector,
            k = default_params["max_top_k_results"]
        )

    # Filter indices using query metadata
    filtered_indices = filter_by_metadata(
        indices = nearest_k_indices,
        metadata_store = metadata_store or [],
        filter_criteria = query_metadata
    )
    
    # Fetch new articles if not enough results found
    if len(filtered_indices) < default_params["max_top_k_results"] // 2 + 1:
        try:
            article_df = run_news_pipeline(
                query_params = news_query_params,
                api_key = news_api_key,
                log_file_path = default_params.get('logs_path'),
                raw_data_path = default_params.get('raw_data_path'),
                sentiment_pipeline = model_pipelines.get('sentiment_pipeline'),
                summarizer_tokenizer = model_pipelines.get('summarizer_tokenizer'),
                summarization_pipeline = model_pipelines.get('summarizer_pipeline'),
                ner_pipeline = model_pipelines.get('ner_pipeline'),
                default_min_text_length = default_params.get('min_text_length', 30),
                default_max_text_length = default_params.get('max_text_length', 150),
                summarize = summarize
            )
        except Exception as e:
            raise RuntimeError(str(e))

        # Prepare and embed new documents
        article_df = drop_post_nlp_columns(article_df)
        documents = article_df.apply(prepare_documents, axis = 1).dropna().tolist()
        faiss_index, metadata_store = build_embeddings(
            embedding_model = model_pipelines['embedding_model'],
            documents = documents
        )

        # Save to vector store then reload merged index
        update_vector_store(
            faiss_index,
            metadata_store,
            default_params["vector_store_path"],
            "faiss.index",
            "metadata.pkl"
        )
        faiss_index, metadata_store = load_vector_store(default_params["vector_store_path"])

        # Search again on merged index
        _, nearest_k_indices = similarity_search(
            faiss_index,
            user_query_vector,
            k = default_params["max_top_k_results"]
        )
        filtered_indices = filter_by_metadata(
            indices = nearest_k_indices,
            metadata_store = metadata_store,
            filter_criteria = query_metadata
        )
    
    # Guard against empty metadata store
    if not metadata_store:
        raise RuntimeError("No articles available. Please adjust your fetch settings and try again.")

    # Build documents for LLM context
    documents = [
        {"page_content": metadata_store[i].get("page_content"), "metadata": metadata_store[i]}
        for i in filtered_indices
    ]

    # Build Context Window
    context_window = context_forming(
        n = 20,
        docs = documents,
        max_context_tokens = 5000,
        avg_chars_per_token = 5
    )
    
    # Generate LLM response
    response = generate_answer(
        user_query,
        context = context_window,
        model = "mistral",
        temperature = 0
    )

    # Build article cards info
    article_cards_info = []
    for i in filtered_indices:
        metadata = metadata_store[i]
        article_cards_info.append({
            'title': metadata['title'],
            'source_name': metadata['source_name'],
            'published_date': metadata['published_date'],
            'sentiment_label': metadata['sentiment_label'],
            'severity': metadata['severity'],
            'entities': (metadata.get('persons') or []) + (metadata.get('locations') or []) + (metadata.get('organizations') or []),
            'url': metadata['url'],
            'summary': metadata['page_content']
        })

    return response, article_cards_info