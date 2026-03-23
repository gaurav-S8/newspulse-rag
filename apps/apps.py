# Import Libraries
import os
import sys
import streamlit as st
from datetime import date, datetime, timedelta

# Append system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Custom Modules
from scripts.utils.env_loader import get_env_var
from scripts.utils.model_loader import load_models
from scripts.utils.config_loader import load_config
from scripts.ui.components import render_entity_pills, render_article_cards
from scripts.nlp.query_handler import extract_query_entities, handle_user_query

# Get API Keys
NEWS_API_KEY = get_env_var("NEWS_API_KEY")

@st.cache_data
def load_cached_config():
    return load_config()

@st.cache_data
def load_params(config):
    return {
        "raw_data_path": config["paths"]["raw_data"],
        "logs_path": config["paths"]["logs"],
        "vector_store_path": config["paths"]["vector_store"],
        "max_text_length": config["defaults"]["max_text_length"],
        "min_text_length": config["defaults"]["min_text_length"],
        "max_top_k_results": config["defaults"]["top_k_results"],
        "ep_everything": config["end_points"]["everything"],
        "ep_top_headlines": config["end_points"]["top-headline"],
        "ep_top_headline_sources": config["end_points"]["top-headline-sources"]
    }

@st.cache_resource
def load_cached_models(config):
    return load_models(config)


def main():
    st.set_page_config(layout = "wide")

    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    if not NEWS_API_KEY:
        st.error("NEWS_API_KEY is not set.")
        st.stop()

    config = load_cached_config()
    default_params = load_params(config)
    model_pipelines = load_cached_models(config)

    with st.sidebar:
        st.markdown("**Fetch Settings**")
        from_date = st.date_input("From Date", value = date.today() - timedelta(days = 7))
        to_date = st.date_input("To Date")
        sort_by = st.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])

        st.divider()

        st.markdown("**Latest Queries**")
        if len(st.session_state.query_history) == 0:
            st.caption("No queries yet.")
        else:
            for idx, item in enumerate(st.session_state.query_history):
                with st.expander(item['query'][:40] + "..."):
                    st.caption(item['response'][:200] + "...")

    st.title("NewsPulse")
    st.subheader("Search and explore news articles semantically")

    user_query = st.text_input("Enter your query", placeholder = "What happened in India today ?")

    if st.button("Search"):
        # From Date cannnpt be in the future!!
        if from_date > date.today():
            st.warning("From Date cannot be in the future.")
            st.stop()
        
        # To Date must be later than From Date!!
        if to_date < from_date:
            st.warning("To Date cannot be before From Date.")
            st.stop()

        # User must enter a valid query!!
        if not user_query.strip():
            st.warning("Please enter a query.")
            st.stop()
        
        query_entities, extracted_entities, keyword = extract_query_entities(
            user_query, model_pipelines.get('ner_pipeline'), model_pipelines.get('keybert_model')
        )
        
        # Render extracted entities
        render_entity_pills(extracted_entities)

        if not extracted_entities:
            st.info("No entities found in your query — showing broadly relevant articles.")
        
        try:
            response, article_cards_info = handle_user_query(
                user_query = user_query,
                news_api_key = NEWS_API_KEY,
                default_params = default_params,
                model_pipelines = model_pipelines,
                from_date = from_date,
                to_date = to_date,
                sort_by = sort_by,
                query_entities = query_entities,
                extracted_entities = extracted_entities,
                keyword = keyword,
                summarize = False
            )
        except RuntimeError as e:
            st.warning(str(e))
            st.stop()

        # Store in query history
        st.session_state.query_history.insert(0, {
            'query': user_query,
            'response': response,
            'cards': article_cards_info
        })
        st.session_state.query_history = st.session_state.query_history[:10]

        # Render LLM response
        st.markdown("### Insight: ")
        st.write(response)

        # Render article cards
        st.divider()
        st.markdown("### Sources Used")
        render_article_cards(article_cards_info)
        
if __name__ == "__main__":
    main()