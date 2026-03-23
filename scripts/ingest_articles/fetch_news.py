# Import Libraries
import os
import json
import requests
import pandas as pd
from typing import Dict
from datetime import datetime

# Import Custom Modules
from scripts.utils.api_logger import log_api_request
from scripts.utils.export_utils import save_fetched_articles

def fetch(
    end_point: str,
    params: Dict,
    api_key: str
) -> Dict:
    """
    Send a GET request to a NewsAPI endpoint.

    Parameters
    ----------
    end_point: str
        Full URL of the NewsAPI endpoint.
    params: Dict
        Dictionary of query parameters.
    api_key: str
        NewsAPI authentication key.

    Returns
    -------
    Dict
        Parsed JSON response from the NewsAPI server.
    """
    params['apikey'] = api_key
    response = requests.get(end_point, params = params)
    return response.json()


def get_everything(
    end_point: str,
    api_key: str,
    log_file_path: str,
    raw_data_path: str,
    q: str = None,
    search_in: str = None,
    sources: str = None,
    domains: str = None,
    exclude_domains: str = None,
    from_date: str = None,
    to_date: str = None,
    language: str = None,
    sort_by: str = 'publishedAt',
    page_size: int = 32,
    page: int = 1
) -> Dict:
    """
    Fetch articles from NewsAPI's everything endpoint using advanced filters.

    Parameters
    ----------
    end_point: str
        Full URL for the NewsAPI everything endpoint.
    api_key: str
        NewsAPI authentication key.
    log_file_path: str
        Path to the CSV file where query metadata will be logged.
    raw_data_path: str
        Directory path where fetched articles will be saved.
    q: str, optional
        Keywords or phrases to search for in articles.
    search_in: str, optional
        Fields to restrict the search to ('title', 'description', 'content', or combinations).
    sources: str, optional
        Comma-separated list of news source identifiers.
    domains: str, optional
        Comma-separated list of domains to include.
    exclude_domains: str, optional
        Comma-separated list of domains to exclude.
    from_date: str, optional
        Start date for articles in ISO 8601 format (YYYY-MM-DD).
    to_date: str, optional
        End date for articles in ISO 8601 format (YYYY-MM-DD).
    language: str, optional
        2-letter language code (e.g., 'en' for English).
    sort_by: str, optional
        Sorting method — 'relevancy', 'popularity', or 'publishedAt'. Defaults to 'publishedAt'.
    page_size: int, optional
        Maximum number of results per page. Defaults to 100.
    page: int, optional
        Page number of results to fetch. Defaults to 1.

    Returns
    -------
    Dict
        Parsed JSON response from NewsAPI.
    """
    
    current_time = datetime.now()
    possible_search_in_values = {
        'title', 'description', 'content',
        'title,description', 'title,content', 'description,title', 'description,content',
        'content,title', 'content,description', 'title,description,content',
        'title,content,description', 'description,title,content', 'description,content,title',
        'content,description,title', 'content,title,description'
    }

    params = {}
    if q is not None:
        params['q'] = q
    if search_in is not None and search_in.lower() in possible_search_in_values:
        params['searchIn'] = search_in.lower()
    if sources is not None:
        params['sources'] = sources
    if domains is not None:
        params['domains'] = domains
    if exclude_domains is not None:
        params['excludeDomains'] = exclude_domains
    if from_date is not None:
        params['from'] = from_date
    if to_date is not None:
        params['to'] = to_date
    if language is not None:
        params['language'] = language
    if sort_by in ['relevancy', 'popularity', 'publishedAt']:
        params['sortBy'] = sort_by
    params['pageSize'] = page_size
    params['page'] = page

    data = fetch(end_point, params, api_key)

    if data['status'] == 'ok':
        query_info = [
            end_point, q, search_in, None, None, sources, domains, exclude_domains,
            from_date, to_date, sort_by, language, data['status'], data['totalResults'],
            current_time.strftime("%Y-%m-%d %H:%M:%S"), current_time.date(), None
        ]
        from_date_ = str(from_date).replace(":", "-").replace(" ", "_")
        to_date_ = str(to_date).replace(":", "-").replace(" ", "_")
        name = f'{raw_data_path}/everything/news_articles_{from_date_}_{to_date_}.json'
        save_fetched_articles(data['articles'], name)
    else:
        query_info = [
            end_point, q, search_in, None, None, sources, domains, exclude_domains,
            from_date, to_date, sort_by, language, data['status'], 0,
            current_time.strftime("%Y-%m-%d %H:%M:%S"), current_time.date(), data.get('message')
        ]
    
    log_api_request(log_file_path, query_info)
    return data


"""
    ---------------------------------------------------------
    ---------------------------------------------------------
    ---------------------------------------------------------
    ---------------------------------------------------------
    ---------------------------------------------------------
    THESE FOLLOWING FUNCTIONS ARE NOT USED AT THE MOMENT!!!!!
    ---------------------------------------------------------
    ---------------------------------------------------------
    ---------------------------------------------------------
    ---------------------------------------------------------
    ---------------------------------------------------------
"""


def get_top_headline_sources(
    end_point: str,
    api_key: str,
    log_file_path: str,
    category: str = "general",
    language: str = "en",
    country: str = None
) -> Dict:
    """
    Fetch available news sources from NewsAPI's top-headlines/sources endpoint.

    Parameters
    ----------
    end_point: str
        Full URL for the NewsAPI top-headlines/sources endpoint.
    api_key: str
        NewsAPI authentication key.
    log_file_path: str
        Path to the CSV file where query metadata will be logged.
    category: str, optional
        News category to filter sources by. Defaults to "general".
    language: str, optional
        Language code to filter sources by. Defaults to "en".
    country: str, optional
        2-letter ISO 3166-1 country code to filter sources by (e.g., "us", "gb").

    Returns
    -------
    Dict
        Parsed JSON response from NewsAPI containing matching sources.
    """
    
    current_time = datetime.now()
    params = {}
    if category is not None:
        params['category'] = category
    if language is not None:
        params['language'] = language
    if country is not None:
        params['country'] = country

    data = fetch(end_point, params, api_key)
    query_info = [
        end_point, None, None, country, category, None, None, None, None, None, None,
        language, data['status'], len(data['sources']),
        current_time.strftime("%Y-%m-%d %H:%M:%S"), current_time.date(), None
    ]
    log_api_request(log_file_path, query_info)
    return data


def get_top_headlines(
    end_point: str,
    api_key: str,
    log_file_path: str,
    raw_data_path: str,
    country: str = None,
    category: str = None,
    sources: str = None,
    language: str = None,
    q: str = None,
    page_size: int = 20,
    page: int = None
) -> Dict:
    
    """
    Fetch top news headlines from NewsAPI's top-headlines endpoint.

    Enforces NewsAPI's rule that the `sources` parameter cannot be combined
    with either `country` or `category`.

    Parameters
    ----------
    end_point: str
        Full URL for the NewsAPI top-headlines endpoint.
    api_key: str
        NewsAPI authentication key.
    log_file_path: str
        Path to the CSV file where query metadata will be logged.
    raw_data_path: str
        Directory path where fetched articles will be saved.
    country: str, optional
        2-letter ISO 3166-1 country code to filter articles by location.
    category: str, optional
        News category (e.g., 'business', 'sports').
    sources: str, optional
        Comma-separated list of news source identifiers.
    language: str, optional
        2-letter language code (e.g., 'en' for English).
    q: str, optional
        Search query string to match in article title and content.
    page_size: int, optional
        Number of results per page. Defaults to 20, max 100.
    page: int, optional
        Page number for paginated results.

    Returns
    -------
    Dict
        Parsed JSON response from NewsAPI, or an error dictionary if
        an invalid parameter combination is detected.
    """
    
    current_time = datetime.now()
    params = {}

    if (country is not None or category is not None) and sources is not None:
        data = {'status': 'error', 'message': "You can't mix Source with the Country or Category params."}
    else:
        if country is not None:
            params['country'] = country
        if category is not None:
            params['category'] = category
        if sources is not None:
            params['sources'] = sources
        if language is not None:
            params['language'] = language
        if q is not None:
            params['q'] = q
        params['pageSize'] = min(page_size, 100)
        if page is not None:
            params['page'] = page
        data = fetch(end_point, params, api_key)

    if data['status'] == 'ok':
        query_info = [
            end_point, q, None, country, category, sources, None, None, None, None, None,
            language, data['status'], data['totalResults'],
            current_time.strftime("%Y-%m-%d %H:%M:%S"), current_time.date(), None
        ]
        name = f'{raw_data_path}/top_headlines/news_articles_{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.json'
        save_fetched_articles(data['articles'], name)
    else:
        query_info = [
            end_point, q, None, country, category, sources, None, None, None, None, None,
            language, data['status'], 0,
            current_time.strftime("%Y-%m-%d %H:%M:%S"), current_time.date(), data.get('message')
        ]

    log_api_request(log_file_path, query_info)
    return data