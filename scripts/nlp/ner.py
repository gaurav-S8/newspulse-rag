# Import Libraries
import re
import pandas as pd
from typing import List, Dict
from transformers import Pipeline

def clean_entities(entities: List[List[str]]) -> List[List[str]]:
    """
    Clean extracted named entities by stripping whitespace and punctuation artifacts.

    Parameters
    ----------
    entities: List[List[str]]
        Per-article list of entity strings extracted by the NER pipeline.

    Returns
    -------
    List[List[str]]
        Cleaned list with leading/trailing whitespace, apostrophes, quotes,
        punctuation and possessive 's removed. Empty strings, subword tokens,
        and duplicates are filtered out.
    """
    cleaned = []
    for article_entities in entities:
        seen = set()
        article_cleaned = []
        for e in article_entities:
            e = e.strip().strip("'\".,!?:)(").strip()
            e = re.sub(r"['\u2018\u2019\u02BC]s$", "", e).strip()
            if e and not e.startswith('##') and e not in seen:
                seen.add(e)
                article_cleaned.append(e)
        cleaned.append(article_cleaned)
    return cleaned

def merge_entity_spans(
    entity_list: List[List]
) -> List[List]:
    """
    Merge token-level entity predictions into complete entity strings for a batch of articles.
    
    Parameters
    ----------
    entity_list: List[List]
        List of articles, where each article is a list of token-level entity predictions
        of the same entity group (e.g., all PER entities across one article).
    
    Returns
    -------
    List[List]
        List of merged entity list per article.
    """

    result = []
    for entities in entity_list:
        entities = sorted(entities, key = lambda x: x["start"])
        
        r = set()
        if(len(entities) > 0):
            current_entity = entities[0]['word']
            for i in range(1, len(entities)):
                word = entities[i]['word']
                if(entities[i]['start'] <= entities[i-1]['end'] + 1):
                    if(word.startswith("##")):
                        current_entity += word.replace("##", "")
                    else:
                        current_entity += " " + word
                else:
                    r.add(current_entity)
                    current_entity = word
            r.add(current_entity)
        result.append(list(r))
    return result

def apply_named_entity_recognition(
    df: pd.DataFrame,
    ner_pipeline: Pipeline
) -> pd.DataFrame:
    """
    Apply Named Entity Recognition (NER) to article summaries using batch inference.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame containing a `summary` column with text data.
    ner_pipeline: Pipeline
        Preloaded Hugging Face NER pipeline.
    
    Returns
    -------
    pandas.DataFrame
        The input DataFrame with three additional columns:
        - `persons`: list of extracted person entities or empty list if none found
        - `locations`: list of extracted location entities or empty list if none found
        - `organizations`: list of extracted organization entities or empty list if none found
    """
    
    if 'summary' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'summary' column.")

    summaries = df['summary'].tolist()
    entity_list = ner_pipeline(
        summaries,
        batch_size = 16
    )
    
    persons, locations, organizations = [], [], []
    for article_entities in entity_list:
        per, loc, org = [], [], []
        for entity in article_entities:
            if(entity['entity_group'] == 'ORG'):
                org.append(entity)
            elif(entity['entity_group'] == 'PER'):
                per.append(entity)
            elif(entity['entity_group'] == 'LOC'):
                loc.append(entity)
        persons.append(per)
        locations.append(loc)
        organizations.append(org)

    persons, locations, organizations = merge_entity_spans(persons), merge_entity_spans(locations), merge_entity_spans(organizations)
    persons, locations, organizations = clean_entities(persons), clean_entities(locations), clean_entities(organizations)
    
    df['persons'] = persons
    df['locations'] = locations
    df['organizations'] = organizations
    return df

def analyze_query(
    query: str,
    ner_pipeline: Pipeline
) -> Dict:
    """
    Extract named entities from a user query.
    
    Parameters
    ----------
    query: str
        User search query.
    ner_pipeline: Pipeline
        Preloaded Hugging Face NER pipeline.
    
    Returns
    -------
    Dict
        Dictionary containing extracted persons, locations, and organizations,
        where each value is a list of entity strings or empty list if no entities found.
    """
    
    entity_list = ner_pipeline([query], batch_size = 1)
    per, loc, org = [], [], []
    
    for entity in entity_list[0]:
        if entity['entity_group'] == 'ORG':
            org.append(entity)
        elif entity['entity_group'] == 'PER':
            per.append(entity)
        elif entity['entity_group'] == 'LOC':
            loc.append(entity)
    
    persons, locations, organizations = merge_entity_spans([per]), merge_entity_spans([loc]), merge_entity_spans([org])
    persons = clean_entities(persons)[0] or []
    locations = clean_entities(locations)[0] or []
    organizations = clean_entities(organizations)[0] or []

    return {
        'persons': persons,
        'locations': locations,
        'organizations': organizations
    }