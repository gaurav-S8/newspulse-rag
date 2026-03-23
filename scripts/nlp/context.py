# Import Libraries
import ollama
from typing import List, Dict

def _token_estimate(
    text: str,
    avg_chars_per_token: int
) -> int:
    """
    Rough token count estimate based on character length.

    Parameters
    ----------
    text: str
        Input text to estimate token count for.
    avg_chars_per_token: int
        Average number of characters per token (typically 5).

    Returns
    -------
    int
        Estimated token count.
    """
    return len(text) // avg_chars_per_token

def context_forming(
    n: int,
    docs: List[Dict],
    max_context_tokens: int,
    avg_chars_per_token: int
) -> str:
    """
    Assemble a context string from retrieved documents

    Handles top-N selection, lightweight article headers, and token limit enforcement.

    Parameters
    ----------
    n: int
        Maximum number of documents to include.
    docs: List[Dict]
        Document store — each entry must have a 'page_content' key.
    max_context_tokens: int
        Maximum number of tokens allowed in the assembled context.
    avg_chars_per_token: int
        Average characters per token used for token estimation (typically 5).

    Returns
    -------
    str
        Formatted context string ready to inject into an LLM prompt.
    """
    
    # Top-N selection
    docs_to_use = docs[:n]
    
    # Concatenate with headers and enforce token limits
    context = ""
    for article_num, doc in enumerate(docs_to_use, start = 1):
        content = doc.get('page_content', '').strip()
        block = f"{content}\n"
        if _token_estimate(context + block, avg_chars_per_token) > max_context_tokens:
            break
        context += block
    return context

def generate_answer(
    user_query: str,
    context: str,
    model: str = "mistral",
    temperature: float = 0.0
) -> str:
    """
    Generate a grounded answer from retrieved context using a local Ollama LLM.

    Parameters
    ----------
    user_query: str
        The user's original question.
    context: str
        Assembled context string from context_forming().
    model: str, optional
        Ollama model name to use for generation. Defaults to "mistral".
    temperature: float, optional
        Sampling temperature. 0.0 for deterministic output.

    Returns
    -------
    str
        Generated answer grounded in the retrieved articles.
    """
    
    if not context.strip():
        return "No relevant articles were found to answer your question."

    prompt = f"""
        You are a News Analyst Assistant. Your job is to answer questions strictly based on the provided news articles.
        Do not use any external knowledge or make assumptions beyond what is written.
        
        Instructions:
        - Answer concisely and directly.
        - If multiple articles are relevant, synthesize them into a coherent response.
        - If the articles do not contain enough information to answer, explicitly say: "The available articles do not contain sufficient information to answer this question."
        - Do not speculate or hallucinate facts.
        - Do not quote article headlines or titles in your response.
        
        Retrieved Articles: {context}
        
        User Question: {user_query}
        
        Answer:
    """
    
    response = ollama.chat(
        model = model,
        messages = [
            {"role": "system", "content": "You are a news analyst assistant."},
            {"role": "user", "content": prompt}
        ],
        options = {
            "temperature": temperature,
            "top_p": 1
        }
    )
    return response["message"]["content"]