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
.stApp {
    background: linear-gradient(to right, #f8f9fa, #eef2f7);
}

.main-card {
    background: white;
    padding: 28px;
    border-radius: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}

.title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #1f2937;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 25px;
}

.answer-card {
    background: #f9fafb;
    padding: 18px;
    border-left: 6px solid #2563eb;
    border-radius: 10px;
    margin-top: 15px;
    font-size: 16px;
    line-height: 1.6;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 22px;
    font-size: 16px;
    border: none;
}
div.stButton > button:hover {
    background-color: #1e40af;
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
