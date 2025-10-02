# app.py — Multilingual Slang/Idiom Explainer (EN/ES/PL/TR)
import os
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.set_page_config(page_title="Slang/Idiom Explainer", page_icon="💬", layout="centered")

# ----------------------------
# 0) Exactly-aligned mini corpus
# ----------------------------
DATA = [
    # 1) Gossip / reveal secret
    {"concept_id":"gossip","lang":"en","phrase":"spill the tea","meaning":"share gossip or reveal a secret","usage":"She spilled the tea about the new launch.","source_url":""},
    {"concept_id":"gossip","lang":"es","phrase":"soltar la sopa","meaning":"contar un secreto o chisme","usage":"Al final soltó la sopa sobre la campaña.","source_url":""},
    {"concept_id":"gossip","lang":"pl","phrase":"puścić farbę","meaning":"zdradzić sekret","usage":"W końcu puścił farbę o projekcie.","source_url":""},
    {"concept_id":"gossip","lang":"tr","phrase":"ağzındaki baklayı çıkarmak","meaning":"sırrı açıklamak; ağzından kaçırmak","usage":"Sonunda ağzındaki baklayı çıkardı.","source_url":""},

    # 2) Daydreaming / head in the clouds
    {"concept_id":"daydream","lang":"en","phrase":"have your head in the clouds","meaning":"be distracted or daydreaming","usage":"He had his head in the clouds during the briefing.","source_url":""},
    {"concept_id":"daydream","lang":"es","phrase":"estar en las nubes","meaning":"estar distraído; soñar despierto","usage":"En clase siempre está en las nubes.","source_url":""},
    {"concept_id":"daydream","lang":"pl","phrase":"bujać w obłokach","meaning":"marzyć; bujać w obłokach","usage":"Na spotkaniu tylko bujał w obłokach.","source_url":""},
    {"concept_id":"daydream","lang":"tr","phrase":"aklı havada olmak","meaning":"dalgın olmak; hayallere dalmak","usage":"Toplantıda aklı tamamen havadaydı.","source_url":""},

    # 3) Low-key / subtly
    {"concept_id":"lowkey","lang":"en","phrase":"low-key","meaning":"subtly; a little; not openly","usage":"I’m low-key excited about this collab.","source_url":""},
    {"concept_id":"lowkey","lang":"es","phrase":"de tranquis","meaning":"de forma discreta; sin alardear","usage":"Lo celebramos de tranquis con el equipo.","source_url":""},
    {"concept_id":"lowkey","lang":"pl","phrase":"po cichu","meaning":"dyskretnie; po cichu","usage":"Zrobili to po cichu, bez ogłoszeń.","source_url":""},
    {"concept_id":"lowkey","lang":"tr","phrase":"çaktırmadan","meaning":"göze batmadan; usulca","usage":"Çaktırmadan birkaç değişiklik yaptık.","source_url":""},

    # 4) Mid / average
    {"concept_id":"mid","lang":"en","phrase":"mid","meaning":"average; not great","usage":"Tbh, the results were mid.","source_url":""},
    {"concept_id":"mid","lang":"es","phrase":"del montón","meaning":"normalito; sin destacar","usage":"Sinceramente, el vídeo quedó del montón.","source_url":""},
    {"concept_id":"mid","lang":"pl","phrase":"takie sobie","meaning":"średnie; nic specjalnego","usage":"Szczerze, wyniki są takie sobie.","source_url":""},
    {"concept_id":"mid","lang":"tr","phrase":"orta karar","meaning":"ortalama; vasat","usage":"Açıkçası performans orta karardı.","source_url":""},

    # 5) Lose it / get very angry
    {"concept_id":"loseit","lang":"en","phrase":"lose it","meaning":"become extremely angry or upset","usage":"I almost lost it when the app crashed.","source_url":""},
    {"concept_id":"loseit","lang":"es","phrase":"perder los papeles","meaning":"perder el control; enfadarse mucho","usage":"Con el retraso, perdió los papeles.","source_url":""},
    {"concept_id":"loseit","lang":"pl","phrase":"puścić nerwy","meaning":"stracić panowanie nad sobą","usage":"Prawie puściły mi nerwy przy tej awarii.","source_url":""},
    {"concept_id":"loseit","lang":"tr","phrase":"kafayı yemek","meaning":"çok sinirlenmek; kendini kaybetmek","usage":"Uygulama çökünce az kalsın kafayı yiyordum.","source_url":""},

    # 6) Get hyped / fired up
    {"concept_id":"hype","lang":"en","phrase":"get hyped","meaning":"become very excited or fired up","usage":"The crowd got hyped before the drop.","source_url":""},
    {"concept_id":"hype","lang":"es","phrase":"venirse arriba","meaning":"animarse mucho; venirse arriba","usage":"Con el tema nuevo todos se vinieron arriba.","source_url":""},
    {"concept_id":"hype","lang":"pl","phrase":"nakręcić się","meaning":"mocno się nakręcić; podekscytować","usage":"Publika szybko się nakręciła.","source_url":""},
    {"concept_id":"hype","lang":"tr","phrase":"gaza gelmek","meaning":"coşmak; hemen motive olmak","usage":"Kalabalık bir anda gaza geldi.","source_url":""},

    # 7) Ghosting
    {"concept_id":"ghosting","lang":"en","phrase":"ghosting","meaning":"suddenly cutting off contact","usage":"After two dates, it was pure ghosting.","source_url":""},
    {"concept_id":"ghosting","lang":"es","phrase":"hacer ghosting","meaning":"dejar de responder sin explicación","usage":"Después del mensaje, me hizo ghosting.","source_url":""},
    {"concept_id":"ghosting","lang":"pl","phrase":"zniknąć bez słowa","meaning":"przestać się odzywać; zniknąć","usage":"Po rozmowie zniknął bez słowa.","source_url":""},
    {"concept_id":"ghosting","lang":"tr","phrase":"ghostlamak / ortadan kaybolmak","meaning":"hiçbir açıklama yapmadan iletişimi kesmek","usage":"İki görüşmeden sonra resmen ghostladı.","source_url":""},

    # 8) Not in the mood / off today
    {"concept_id":"offday","lang":"en","phrase":"not in the mood","meaning":"feeling off; not up for it","usage":"I’m not in the mood for calls today.","source_url":""},
    {"concept_id":"offday","lang":"es","phrase":"no tener el día","meaning":"estar regular; no estar de humor","usage":"Hoy no tengo el día para reuniones.","source_url":""},
    {"concept_id":"offday","lang":"pl","phrase":"nie w sosie","meaning":"mieć zły humor; być nie w formie","usage":"Jestem dziś nie w sosie.","source_url":""},
    {"concept_id":"offday","lang":"tr","phrase":"keyfi yok","meaning":"modu düşük; canı istemiyor","usage":"Bugün pek keyfim yok toplantılara.","source_url":""},
]

DF = pd.DataFrame(DATA)
DF["blob"] = DF["phrase"] + " — " + DF["meaning"] + " — " + DF["usage"]

# -----------------------------------------
# 1) Cache models to speed reloads
# -----------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedder_and_embeddings(df: pd.DataFrame):
    emb_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vecs = emb_model.encode(df["blob"].tolist(), normalize_embeddings=True)
    return emb_model, vecs

@st.cache_resource(show_spinner=False)
def load_generator():
    candidates = ["google/gemma-3-270m", "google/flan-t5-small"]  # prefer Gemma
    for mid in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(mid, token=os.environ.get("HUGGING_FACE_HUB_TOKEN", True))
            if "t5" in mid.lower():
                gen = pipeline("text2text-generation", model=mid, tokenizer=tok)
                return gen, mid, "t2t"
            else:
                mdl = AutoModelForCausalLM.from_pretrained(mid, token=os.environ.get("HUGGING_FACE_HUB_TOKEN", True))
                gen = pipeline("text-generation", model=mdl, tokenizer=tok)
                return gen, mid, "causal"
        except Exception as e:
            print(f"[warn] generator load failed for {mid}: {e}")
    raise RuntimeError("No generator model could be loaded.")

EMB_MODEL, EMBEDDINGS = load_embedder_and_embeddings(DF)
GEN, GEN_ID, GEN_KIND = load_generator()

STYLES = {
    "learner": "Explain simply for language learners. Avoid slang in the explanation and include ONE short example.",
    "casual":  "Use a casual, friendly tone and keep it short.",
    "formal":  "Use a clear, formal, brand-safe tone suitable for documentation."
}

# -----------------------
# 2) Core helper methods
# -----------------------
def search(query: str, k: int = 3) -> pd.DataFrame:
    qv = EMB_MODEL.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(qv.reshape(1, -1), EMBEDDINGS)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    hits = DF.iloc[top_idx].copy()
    hits["score"] = sims[top_idx]
    return hits[["lang","phrase","meaning","usage","source_url","score"]]

def generate_explanation(query: str, style: str, hits: pd.DataFrame) -> str:
    ctx = "\n".join([f"- [{r.lang}] {r.phrase}: {r.meaning} (e.g., {r.usage})" for _, r in hits.iterrows()])
    instr = STYLES.get(style, STYLES["learner"])
    prompt = (
        f"Explain the expression '{query}'. {instr}\n"
        f"Use the retrieved examples below as context and mention the language code in examples.\n"
        f"Retrieved examples:\n{ctx}\n\nAnswer:"
    )
    if GEN_KIND == "t2t":
        return GEN(prompt, max_new_tokens=140, do_sample=False)[0]["generated_text"]
    else:
        return GEN(prompt, max_new_tokens=140, do_sample=False)[0]["generated_text"]

# ---------------
# 3) Streamlit UI
# ---------------
st.title("💬 Slang / Idiom Explainer — Multilingual PoC")
st.caption(f"Embedding: paraphrase-multilingual-MiniLM • Generator: {GEN_ID}")

col1, col2 = st.columns([3, 2])
with col1:
    query = st.text_input("Enter a slang/idiom (any of EN/ES/PL/TR)", value="")
with col2:
    style = st.selectbox("Style", options=["learner", "casual", "formal"], index=0)

k = st.slider("How many examples to retrieve (k)?", 1, 5, 3)

if st.button("Explain"):
    if not query.strip():
        st.warning("Please enter a slang/idiom.")
        st.stop()

    # Nearest neighbours (semantic retrieval)
    hits = search(query.strip(), k=k)
    st.subheader("Top retrieved examples")
    st.dataframe(hits, use_container_width=True)

    # 🔹 NEW: exact aligned equivalents by concept_id
    if not hits.empty:
        # match top hit back to DF to read its concept_id
        top_phrase, top_lang = hits.iloc[0]["phrase"], hits.iloc[0]["lang"]
        match = DF[(DF["phrase"] == top_phrase) & (DF["lang"] == top_lang)]
        if not match.empty:
            top_concept = match.iloc[0]["concept_id"]
            aligned = DF[DF["concept_id"] == top_concept][["lang", "phrase", "meaning", "usage", "source_url"]]
            st.subheader("Aligned equivalents (exact cross-language matches)")
            st.dataframe(aligned, use_container_width=True)

    st.subheader("Explanation")
    with st.spinner("Generating..."):
        answer = generate_explanation(query.strip(), style, hits)
    st.write(answer)

st.divider()
with st.expander("Show demo corpus"):
    st.dataframe(DF[["concept_id","lang","phrase","meaning","usage","source_url"]], use_container_width=True)
