import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ================= API KEY =================
os.environ["GOOGLE_API_KEY"] = "AIzaSyCnmlWHnf-f3DxyPHmN4Qa8KB9M6_1dYQM"


# ================= STREAMLIT =================
st.set_page_config(page_title="Railway Passenger Explainer")
st.title("🚆 Railway Passenger Process Explainer Bot")


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
        model="gemini-3-flash-preview",
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
question = st.text_input(
    "Ask a railway-related question:",
    placeholder="e.g. What are the rules for changing boarding point?"
)

if st.button("Ask"):
    if question.strip():
        with st.spinner("Fetching answer from documents..."):
            answer = rag_chain.invoke(question)
            st.success("Answer")
            st.write(answer)
    else:
        st.warning("Please enter a question.")
