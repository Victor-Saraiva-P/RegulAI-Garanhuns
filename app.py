import streamlit as st
from config import *
from db import load_or_fetch_documents
from embeddings import create_embeddings, split_documents, create_or_load_vectorstore
from llm import init_llm, rag_search

# Configura a página do Streamlit
st.set_page_config(
    page_title="RegulAI - Chatbot de Leis Municipais de Garanhuns",
    page_icon="📜",
    layout="centered"
)

st.title("📜 RegulAI - Chatbot de Leis Municipais de Garanhuns 🏛️")
st.write("Pergunte sobre leis municipais e receba respostas baseadas nos textos legais!")

# Carrega documentos (do cache ou do MongoDB, dependendo da disponibilidade)
raw_documents = load_or_fetch_documents()
if not raw_documents:
    st.error("Nenhum documento válido encontrado. Encerrando execução.")
    st.stop()

# Cria ou carrega modelo de embeddings
embeddings = create_embeddings()

# Divide documentos em pedaços (chunks)
split_docs = split_documents(raw_documents)
if not split_docs:
    st.error("Nenhum pedaço de documento encontrado para FAISS.")
    st.stop()

# Cria ou carrega o índice vetorial (FAISS)
vector_store = create_or_load_vectorstore(split_docs, embeddings)

# Inicializa a LLM (Groq)
llm = init_llm()

st.subheader("💬 Faça sua pergunta:")
user_input = st.text_input("Digite sua pergunta:")

# Se o usuário inserir uma pergunta, faz a busca via RAG e exibe a resposta
if user_input:
    with st.spinner("Buscando resposta..."):
        response = rag_search(user_input, vector_store, llm)
        st.text_area("Resposta:", response, height=200)
