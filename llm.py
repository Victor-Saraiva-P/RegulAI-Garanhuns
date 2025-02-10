# llm.py
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import os

# Carrega as variáveis do arquivo .env
load_dotenv()
groqModel = os.environ.get('GROQ_MODEL')

def init_llm():
    print("Configurando LLM Groq...\n")
    llm = ChatGroq(model=groqModel)
    print("LLM configurado!")
    return llm

def rag_search(query: str, vector_store, llm):
    print(f"\nRealizando busca para: {query}")

    # Busca vetorial via FAISS
    relevant_docs = vector_store.similarity_search(query, k=5)
    print(f"Documentos relevantes encontrados: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs):
        metadata = doc.metadata  # Obtendo os metadados do chunk
        print(f"📜 **Documento {i+1}**")
        print(f"   - 🏛 **Lei:** {metadata.get('numero_lei', 'Desconhecido')}/{metadata.get('ano_lei', 'Desconhecido')}")
        print(f"   - 📖 **Ementa:** {metadata.get('ementa', 'Sem ementa')}")
        print(f"   - 📄 **Trecho:** {doc.page_content[:200]}...\n")  # Mostrando só os primeiros 200 caracteres


    # Concatena os trechos para formar o contexto
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = (
        f"Baseando-se no texto das leis a seguir, responda com precisão:\n\n"
        f"{context}\n\nPergunta: {query}\nResposta:"
    )
    print("Enviando prompt para LLM...")
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        st.error(f"Erro ao invocar LLM: {e}")
        response = "Desculpe, ocorreu um erro ao gerar a resposta."
    print("Resposta gerada!")
    return response
