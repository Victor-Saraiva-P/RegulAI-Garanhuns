# llm.py
import streamlit as st
from langchain_groq import ChatGroq

def init_llm():
    print("Configurando LLM Groq...\n")
    llm = ChatGroq(model="llama3-8b-8192")
    print("LLM configurado!")
    return llm

def rag_search(query: str, vector_store, llm):
    print(f"\nRealizando busca para: {query}")

    # Busca vetorial via FAISS
    relevant_docs = vector_store.similarity_search(query, k=5)
    print(f"Documentos relevantes encontrados: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs):
        print(f"Documento {i+1} - Início do texto:\n{doc.page_content[:200]}...\n")

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
