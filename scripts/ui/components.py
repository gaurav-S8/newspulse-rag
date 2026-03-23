# Import Libraries
import streamlit as st
from typing import List, Dict

SENTIMENT_COLORS = {
    "Positive": "#22C55E",
    "Negative": "#EF4444",
    "Neutral": "#EAB308"
}

SENTIMENT_EMOJIS = {
    "Positive": "🟢",
    "Negative": "🔴",
    "Neutral": "🟡"
}

def render_entity_pills(
    entities: List[str]
) -> None:
    """
    Render extracted query entities as coloured HTML pills.

    Parameters
    ----------
    entities: List[str]
        List of entity strings to render.

    Returns
    -------
    None
    """
    if not entities:
        return
    entity_html = " ".join([
        f'<span style="background-color:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:10px; font-size:11px; margin:2px">{e}</span>'
        for e in entities[:6]
    ])
    st.markdown(
        f'<span style="color:#EF4444; font-weight:700">Extracted Entities</span> → {entity_html}',
        unsafe_allow_html = True
    )


def render_article_cards(
    article_cards_info: List[Dict]
) -> None:
    """
    Render article cards in a two-column grid layout.

    Parameters
    ----------
    article_cards_info: List[Dict]
        List of article metadata dicts containing title, url, source_name,
        published_date, sentiment_label, severity, entities, and summary.

    Returns
    -------
    None
    """
    cols = st.columns(2)
    for idx, card in enumerate(article_cards_info):
        with cols[idx % 2]:
            sentiment_color = SENTIMENT_COLORS.get(card['sentiment_label'], "#94A3B8")
            sentiment_emoji = SENTIMENT_EMOJIS.get(card['sentiment_label'], "⚪")

            entity_html = " ".join([
                f'<span style="background-color:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:10px; font-size:11px; margin:2px">{e}</span>'
                for e in card['entities'][:6]
            ])

            st.markdown(f"""
                <div style="border:1px solid #E2E8F0; border-radius:10px; overflow:hidden; margin-bottom:12px">
                    <div style="background:#F8FAFC; padding:8px 14px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #E2E8F0">
                        <span style="font-size:12px; color:#64748B">📰 {card['source_name']} &nbsp;·&nbsp; 📅 {card['published_date']}</span>
                        <span style="font-size:12px; font-weight:600; color:{sentiment_color}">{sentiment_emoji} {card['sentiment_label']}</span>
                    </div>
                    <div style="padding:12px 14px">
                        <a href="{card['url']}" target="_blank" style="font-size:14px; font-weight:600; color:#1E293B; text-decoration:none">{card['title']}</a>
                        <p>{card['summary']}</p>
                        <div style="margin-top:10px">{entity_html}</div>
                    </div>
                </div>
                """, unsafe_allow_html = True
            )