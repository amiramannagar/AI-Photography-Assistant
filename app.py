from rag import ask_rag
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Photography Assistant",
    page_icon="📸",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>

header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container{
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e3a8a);
    color:white;
}

.title{
    text-align:center;
    font-size:48px;
    font-weight:700;
    color:#60a5fa;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:35px;
}

.card{
    background:rgba(255,255,255,0.08);
    padding:25px;
    border-radius:18px;
    border:1px solid rgba(255,255,255,0.08);
}

.stButton>button{
    background:#111827;
    color:white;
    border-radius:10px;
    padding:10px 25px;
    border:1px solid #334155;
    font-weight:600;
}

.stButton>button:hover{
    border:1px solid #60a5fa;
    color:#60a5fa;
}

.footer{
    text-align:center;
    margin-top:50px;
    color:#93c5fd;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown("<div class='title'>📸 AI Photography Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Get the best camera settings using AI + RAG</div>", unsafe_allow_html=True)


# =========================
# INPUT
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

question = st.text_area(
    "📷 Your Question",
    placeholder="Best settings for Golden Hour photography",
    height=130
)

generate = st.button("Generate Settings 🚀")

st.markdown("</div>", unsafe_allow_html=True)


# =========================
# OUTPUT
# =========================
if generate and question:

    with st.spinner("Generating best camera settings... 📸"):
        result = ask_rag(question)

    settings = result["settings"]

    iso = settings["iso"]
    aperture = settings["aperture"]
    shutter = settings["shutter_speed"]

    st.markdown("## 📊 Recommended Setup")

    c1, c2, c3 = st.columns(3)

    c1.metric("ISO", iso if iso else "N/A")
    c2.metric("Aperture", aperture if aperture else "N/A")
    c3.metric("Shutter Speed", shutter if shutter else "N/A")


# =========================
# FOOTER
# =========================
st.markdown(
    "<div class='footer'>Made with 💙 by <b>Amira Mahmoud</b></div>",
    unsafe_allow_html=True
)