import os
import re
import pickle
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Carrega as variáveis do arquivo .env
load_dotenv()
chunk_size = os.environ.get('CHUNK_SIZE')
chunk_overlap = os.environ.get('CHUNK_OVERLAP')
embeddingModel = os.environ.get('EMBEDDING_MODEL')


# Criando diretório para cache
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Caminhos para cache
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "cached_embeddings.pkl")
SPLIT_DOCS_PATH = os.path.join(CACHE_DIR, "cached_split_docs.pkl")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")

def extract_metadata(text):
    """Extrai número da lei, ano e ementa do texto usando expressões regulares."""
    
    # Expressão para capturar "LEI Nº XXXX/AAAA"
    numero_lei_match = re.search(r"LEI\s+N[ºo]\s+(\d{3,5})/(\d{4})", text, re.IGNORECASE)
    numero_lei = numero_lei_match.group(1) if numero_lei_match else "Desconhecido"
    ano_lei = numero_lei_match.group(2) if numero_lei_match else "Desconhecido"

    # Expressão para capturar "EMENTA: ..."
    ementa_match = re.search(r"EMENTA:\s*(.*)", text, re.IGNORECASE)
    ementa = ementa_match.group(1).strip() if ementa_match else "Sem ementa"

    return {
        "numero_lei": numero_lei,
        "ano_lei": ano_lei,
        "ementa": ementa
    }

def create_embeddings():
    """Cria ou carrega os embeddings do cache."""
    if os.path.exists(EMBEDDINGS_PATH):
        print("Carregando embeddings do cache...")
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("Gerando embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name=embeddingModel)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)
        print("Embeddings criados e salvos!")

    return embeddings

def split_documents(raw_documents):
    """Divide documentos e adiciona metadados"""
    if os.path.exists(SPLIT_DOCS_PATH):
        print("Carregando documentos divididos do cache...")
        with open(SPLIT_DOCS_PATH, "rb") as f:
            split_docs = pickle.load(f)
    else:
        print("Dividindo documentos...")

        docs = []  # ✅ Inicializando a lista de documentos

        for doc in raw_documents:
            metadata = extract_metadata(doc)  # Extraindo metadados
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
            chunks = text_splitter.split_text(doc)

            for chunk in chunks:
                docs.append(Document(page_content=chunk, metadata=metadata))

        # Salvar para evitar reprocessamento
        with open(SPLIT_DOCS_PATH, "wb") as f:
            pickle.dump(docs, f)

        print(f"Total de pedaços gerados: {len(docs)}")

        split_docs = docs  # ✅ Garantindo que split_docs tenha um valor válido

    return split_docs


def create_or_load_vectorstore(split_docs, embeddings):
    """Carrega FAISS se existir, senão cria e salva."""
    if os.path.exists(FAISS_INDEX_PATH):
        print("Carregando FAISS do disco...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS carregado com sucesso!")
    else:
        print("Criando base vetorial FAISS...")
        metadatas = [{"numero_lei": doc.metadata["numero_lei"],
                      "ano_lei": doc.metadata["ano_lei"],
                      "ementa": doc.metadata["ementa"]} for doc in split_docs]

        for doc, metadata in zip(split_docs, metadatas):
            doc.metadata = metadata  # Atribuindo os metadados diretamente nos documentos

        vector_store = FAISS.from_documents(split_docs, embeddings)  # Agora sem 'metadatas'

        vector_store.save_local(FAISS_INDEX_PATH)
        print("Base vetorial FAISS criada e salva com sucesso!")

    return vector_store
