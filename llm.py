import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
groqModel = os.environ.get('GROQ_MODEL')

def init_llm():
    """
    Inicializa a LLM (ChatGroq) com o modelo definido em .env.
    Retorna o objeto de modelo configurado.
    """
    print("Configurando LLM Groq...\n")
    llm = ChatGroq(model=groqModel)
    print("LLM configurado!")
    return llm

def rag_search(query: str, vector_store, llm):
    """
    Faz uma busca RAG (Retrieval-Augmented Generation) utilizando:
      - busca exata por número de lei, se presente na query;
      - caso contrário, busca vetorial no FAISS.
    Em seguida, constrói um prompt com contexto e invoca a LLM.
    Retorna o texto de resposta gerado.
    """
    # Regex para identificar número de lei no formato 'X.XXX/AAAA' ou 'XXXX/AAAA'
    lei_match = re.search(r"\b(\d{1,5}(?:\.\d{1,5})?/\d{4})\b", query)
    if lei_match:
        raw_numero = lei_match.group(1)
        # Remove pontos para corresponder ao que está salvo nos metadados
        numero_lei = raw_numero.replace(".", "")
        print(f"Usuário perguntou especificamente pela lei: {numero_lei}")

        # Filtra chunks cujo metadata tenha 'numero_lei' igual ao que foi capturado
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
        relevant_docs = vector_store.similarity_search(query, k=5)

    print(f"Documentos relevantes: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs):
        metadata = doc.metadata
        print(f"📜 Doc {i+1} - Lei: {metadata.get('numero_lei')}")
        print(f"   Trecho: {doc.page_content[:200]}...\n")

    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Instruções de resposta para a LLM
    prompt = (
        f"Você é um especialista jurídico encarregado de interpretar leis municipais de Garanhuns. "
        f"Seu objetivo é fornecer respostas claras, precisas e diretas para perguntas sobre cada lei. "
        f"Sempre baseie suas respostas exclusivamente no texto da lei fornecida, sem suposições externas. "
        f"Se uma pergunta mencionar um número de lei, verifique se ele está presente no texto antes de responder. "
        f"\n\nRegras para resposta:"
        f"\n- Se houver uma data mencionada, forneça exatamente essa data sem alterar o formato."
        f"\n- Se houver um nome próprio, transcreva-o exatamente como aparece no texto da lei."
        f"\n- Se houver um valor financeiro, forneça exatamente o valor sem reescrevê-lo ou arredondá-lo."
        f"\n- Se a informação não estiver no texto, responda explicitamente que o dado não foi encontrado."
        f"\n- Mantenha a resposta objetiva e sem repetições desnecessárias."
        f"\n- Se a pergunta tiver um formato Existe alguma lei? Sempre cite a lei."
        f"\n\n{context}\n\nPergunta: {query}\nResposta:"
    )

    # Chamada ao modelo
    try:
        print("Enviando prompt para LLM...")
        response_ai_message = llm.invoke(prompt)
        response_text = response_ai_message.content
        print("Resposta gerada!")
    except Exception as e:
        response_text = f"Erro ao invocar LLM: {e}"

    return response_text

