# app.py
import streamlit as st
from config import *  # Carrega variáveis e configurações
from db import get_mongo_collection, load_documents
from embeddings import create_embeddings, split_documents, create_vectorstore
from llm import init_llm, rag_search
from pyngrok import ngrok
import os

# Configuração do layout da página
st.set_page_config(page_title="RegulAI - Chatbot de Leis Municipais de Garanhuns", page_icon="📜", layout="centered")
st.title("📜 RegulAI - Chatbot de Leis Municipais de Garanhuns 🏛️")
st.write("Pergunte sobre leis municipais e receba respostas baseadas nos textos legais!")

# Conectar ao MongoDB e carregar documentos
collection = get_mongo_collection()
test_doc = collection.find_one()
if not test_doc:
    st.error("Nenhum documento encontrado no MongoDB! Verifique a conexão e os dados.")
    st.stop()

raw_documents = load_documents(collection)
if not raw_documents:
    st.error("Nenhum documento válido encontrado. Encerrando execução.")
    st.stop()

# Criação do vectorstore
embeddings = create_embeddings()
split_docs = split_documents(raw_documents)
if not split_docs:
    st.error("Nenhum pedaço de documento encontrado para FAISS. Verifique a extração de dados.")
    st.stop()
vector_store = create_vectorstore(split_docs, embeddings)

# Configuração do LLM
llm = init_llm()

#Interface
st.subheader("💬 Faça sua pergunta:")
user_input = st.text_input("Digite sua pergunta:")
if user_input:
    with st.spinner("Buscando resposta..."):
        response = rag_search(user_input, vector_store, llm)
        content = response.content
        # Aqui, exiba somente a resposta limpa para o usuário:
        st.text_area("Resposta:", content, height=200)
