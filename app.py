# app.py — tiny local Streamlit stub for the Slang Explainer
import streamlit as st
import pandas as pd

DATA = [
    {"phrase":"spill the tea","meaning":"share the gossip or secret","usage":"She spilled the tea about the new launch.","source_url":"https://example.com"},
    {"phrase":"low-key","meaning":"subtly; a little bit","usage":"I’m low-key excited about this collab.","source_url":"https://example.com"},
    {"phrase":"ghosting","meaning":"suddenly cutting off all communication","usage":"He stopped replying—total ghosting.","source_url":"https://example.com"},
    {"phrase":"mid","meaning":"average; not great","usage":"The results were mid tbh.","source_url":"https://example.com"},
]
df = pd.DataFrame(DATA)

st.title("Slang/Idiom Explainer — PoC")
q = st.text_input("Enter a slang/idiom (e.g., 'spill the tea')")

def simple_search(query: str, k: int = 3):
    if not query:
        return df.head(k).copy()
    ql = query.lower()
    # very simple score: token overlap with phrase + usage
    def score(row):
        text = f"{row['phrase']} {row['usage']}".lower()
        return sum(tok in text for tok in ql.split())
    hits = df.copy()
    hits["score"] = hits.apply(score, axis=1)
    hits = hits.sort_values("score", ascending=False).head(k)
    return hits[["phrase","meaning","usage","source_url","score"]]

if q:
    st.subheader("Top examples")
    st.dataframe(simple_search(q))
else:
    st.caption("Type a slang term to see examples.")
