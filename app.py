import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ================= API KEY =================
# ⚠️ RECOMMENDED: move this to Streamlit Secrets later
os.environ["GOOGLE_API_KEY"] = "AIzaSyCnmlWHnf-f3DxyPHmN4Qa8KB9M6_1dYQM"


# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="Railway Passenger Explainer",
    page_icon="🚆",
    layout="centered"
)

# ================= CUSTOM HTML + CSS =================
st.markdown("""
<style>

/* ========== BACKGROUND LIGHTS ========== */
.stApp {
    background:
        radial-gradient(circle at top left, rgba(99,102,241,0.18), transparent 40%),
        radial-gradient(circle at bottom right, rgba(14,165,233,0.18), transparent 40%),
        linear-gradient(to right, #f8fafc, #eef2ff);
}

/* ========== MAIN GLASS CARD ========== */
.main-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(18px);
    padding: 36px;
    border-radius: 26px;
    box-shadow:
        0 30px 65px rgba(0,0,0,0.18),
        inset 0 1px 0 rgba(255,255,255,0.7);
    border: 1px solid rgba(255,255,255,0.45);
    animation: slideFade 0.8s ease;
}

/* Hover glow */
.main-card:hover {
    box-shadow:
        0 35px 75px rgba(99,102,241,0.35),
        0 0 45px rgba(99,102,241,0.30);
}

/* ========== TITLE ========== */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 900;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glowText 2.5s infinite alternate;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #475569;
    font-size: 16px;
    margin: 12px 0 30px;
}

/* ========== INPUT ========== */
input {
    border-radius: 18px !important;
    padding: 15px !important;
    background: rgba(255,255,255,0.9) !important;
    box-shadow: inset 0 3px 10px rgba(0,0,0,0.1);
}

/* ========== BUTTON ========== */
div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border-radius: 18px;
    padding: 14px 30px;
    font-size: 16px;
    font-weight: 700;
    border: none;
    box-shadow: 0 15px 35px rgba(124,58,237,0.45);
    animation: pulse 2.2s infinite;
}

div.stButton > button:hover {
    transform: translateY(-4px);
    box-shadow:
        0 20px 45px rgba(124,58,237,0.6),
        0 0 35px rgba(124,58,237,0.65);
}

/* ========== ANSWER CARD ========== */
.answer-card {
    background: rgba(248,250,252,0.98);
    padding: 24px;
    border-radius: 20px;
    margin-top: 26px;
    font-size: 16px;
    line-height: 1.8;
    border-left: 6px solid #6366f1;
    box-shadow: 0 18px 40px rgba(0,0,0,0.2);
    animation: fadeUp 0.6s ease;
}

/* ========== ANIMATIONS ========== */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideFade {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes glowText {
    from { text-shadow: 0 0 10px rgba(124,58,237,0.4); }
    to { text-shadow: 0 0 22px rgba(124,58,237,0.9); }
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(124,58,237,0.45); }
    70% { box-shadow: 0 0 0 18px rgba(124,58,237,0); }
    100% { box-shadow: 0 0 0 0 rgba(124,58,237,0); }
}

</style>
""", unsafe_allow_html=True)


# ================= CACHE: VECTOR DB =================
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )
    return db.as_retriever(search_kwargs={"k": 4})


# ================= CACHE: LLM =================
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",   # safer model
        temperature=0.2
    )


retriever = load_vector_db()
llm = load_llm()


# ================= PROMPT =================
prompt = PromptTemplate(
    template="""
You are a Railway Passenger Process Explainer Bot.

Answer ONLY using the provided context.
Do NOT book tickets.
Do NOT give live schedules.

If the answer is not found, say:
"Information not available in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# ================= RAG CHAIN =================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ================= UI =================
st.markdown("""
<div class="main-card">
    <div class="title">🚆 Railway Passenger Process Explainer</div>
    <div class="subtitle">
        Clear answers generated only from official railway documents
    </div>
""", unsafe_allow_html=True)

st.markdown("#### 💬 Ask your question")

question = st.text_input(
    "",
    placeholder="e.g. What are the rules for changing boarding point?"
)

if st.button("Ask"):
    if question.strip():
        with st.spinner("🔍 Fetching answer from railway documents..."):
            answer = rag_chain.invoke(question)

        st.markdown(f"""
        <div class="answer-card">
            {answer}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a question.")

st.markdown("</div>", unsafe_allow_html=True)


# ================= SIDEBAR =================
st.sidebar.markdown("""
### ℹ️ About
This AI assistant explains **railway passenger processes**
using document-based retrieval (RAG).

❌ No ticket booking  
❌ No live schedules  
✅ Official documents only  

---
### 🛠 Tech Stack
- Streamlit  
- LangChain  
- ChromaDB  
- Gemini API  
""")

# ================= FOOTER =================
st.markdown("""
<hr>
<p style="text-align:center; color:gray; font-size:13px;">
Built with ❤️ using Streamlit & Gemini
</p>
""", unsafe_allow_html=True)
