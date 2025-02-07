import streamlit as st
from config import *  # Carrega variáveis e configurações
from db import load_or_fetch_documents
from embeddings import create_embeddings, split_documents, create_or_load_vectorstore
from llm import init_llm, rag_search
from pyngrok import ngrok
import os

# Configuração do layout da página
st.set_page_config(page_title="RegulAI - Chatbot de Leis Municipais de Garanhuns", page_icon="📜", layout="centered")
st.title("📜 RegulAI - Chatbot de Leis Municipais de Garanhuns 🏛️")
st.write("Pergunte sobre leis municipais e receba respostas baseadas nos textos legais!")

# Carregar documentos (cache ou MongoDB apenas na primeira vez)
raw_documents = load_or_fetch_documents()
if not raw_documents:
    st.error("Nenhum documento válido encontrado. Encerrando execução.")
    st.stop()

# Criando embeddings e vetores APENAS se necessário
embeddings = create_embeddings()
split_docs = split_documents(raw_documents)
if not split_docs:
    st.error("Nenhum pedaço de documento encontrado para FAISS. Verifique a extração de dados.")
    st.stop()

vector_store = create_or_load_vectorstore(split_docs, embeddings)

# Configuração do LLM
llm = init_llm()

# Interface
st.subheader("💬 Faça sua pergunta:")
user_input = st.text_input("Digite sua pergunta:")

if user_input:
    with st.spinner("Buscando resposta..."):
        response = rag_search(user_input, vector_store, llm)
        content = response.content
        st.text_area("Resposta:", content, height=200)
