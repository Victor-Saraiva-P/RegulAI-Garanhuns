# llm.py
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import os
import re

# Carrega as variáveis do arquivo .env
load_dotenv()
groqModel = os.environ.get('GROQ_MODEL')

def init_llm():
    print("Configurando LLM Groq...\n")
    llm = ChatGroq(model=groqModel)
    print("LLM configurado!")
    return llm


def rag_search(query: str, vector_store, llm):
    # Ajuste da regex:
    #  -> (\d{1,5}(?:\.\d{1,5})?/\d{4}) permite um ou mais pontos entre dígitos
    #  -> Exemplo: "5.275/2024", "5275/2024", "12.345/2024", etc.
    lei_match = re.search(r"\b(\d{1,5}(?:\.\d{1,5})?/\d{4})\b", query)
    if lei_match:
        raw_numero = lei_match.group(1)  # Ex.: "5.275/2024"
        # Remove o(s) ponto(s) para coincidir com o que está salvo nos metadados ("5275/2024")
        numero_lei = raw_numero.replace(".", "")  # "5275/2024"
        print(f"Usuário perguntou especificamente pela lei: {numero_lei}")

        # Filtra diretamente pelo metadado "numero_lei" == "5275/2024"
        matched_chunks = [
            doc for doc in vector_store.docstore._dict.values()
            if doc.metadata.get("numero_lei") == numero_lei
        ]

        if matched_chunks:
            print(f"Encontramos {len(matched_chunks)} chunks com esta lei.")
            relevant_docs = matched_chunks
        else:
            print("Não encontramos um match exato. Indo para busca vetorial...")
            relevant_docs = vector_store.similarity_search(query, k=5)
    else:
        # Se não for uma pergunta específica de lei, busca normal
        relevant_docs = vector_store.similarity_search(query, k=5)

    print(f"Documentos relevantes: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs):
        metadata = doc.metadata
        print(f"📜 **Doc {i+1}** - Lei: {metadata.get('numero_lei')}")
        print(f"   Trecho: {doc.page_content[:200]}...\n")

    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = (
        f"Baseando-se no texto das leis a seguir, responda com precisão:\n\n"
        f"{context}\n\nPergunta: {query}\nResposta:"
    )

    try:
        print("Enviando prompt para LLM...")
        response = llm.invoke(prompt)
        print("Resposta gerada!")
    except Exception as e:
        response = f"Erro ao invocar LLM: {e}"

    return response

