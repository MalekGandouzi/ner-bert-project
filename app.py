import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/ner_improved"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# CSS MODERNE
# =========================
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #2c3e50;
}
.subtitle {
    font-size: 18px;
    color: #6c757d;
    margin-bottom: 30px;
}
.entity {
    padding: 6px 12px;
    margin: 4px;
    border-radius: 20px;
    display: inline-block;
    font-weight: 600;
    font-size: 14px;
}
.PER { background-color: #e74c3c; color: white; }
.ORG { background-color: #3498db; color: white; }
.LOC { background-color: #27ae60; color: white; }
.MISC { background-color: #f39c12; color: white; }
.O { background-color: #ecf0f1; color: #7f8c8d; }
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<div class="main-title">🧠 Reconnaissance d’Entités Nommées</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modèle BERT fine-tuné – Interface IA interactive</div>', unsafe_allow_html=True)

# =========================
# INPUT CARD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
text = st.text_area(
    "✍️ Entrez un texte à analyser",
    height=120,
    placeholder="Exemple : Malek Gandouzi étudie à Polytech Monastir en Tunisie."
)
analyze = st.button("🚀 Lancer l’analyse")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# INFERENCE
# =========================
def predict(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    words = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    labels = [model.config.id2label[p] for p in predictions]
    return words, labels

# =========================
# DISPLAY RESULTS
# =========================
if analyze and text.strip():
    words, labels = predict(text)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Entités détectées")

    html = ""
    for word, label in zip(words, labels):
        if word.startswith("##"):
            html = html.rstrip("</span>")
            html += word[2:] + "</span>"
        else:
            clean_label = label.split("-")[-1]
            html += f'<span class="entity {clean_label}">{word}</span> '

    st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # SUMMARY
    # =========================
    summary = {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}
    for l in labels:
        if l != "O":
            summary[l.split("-")[-1]] += 1

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Résumé")
    for k, v in summary.items():
        st.write(f"**{k}** : {v}")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown(
    "<center style='color:#95a5a6;'>Projet IA & NLP • BERT • Streamlit</center>",
    unsafe_allow_html=True
)
