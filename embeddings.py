# embedding.py
import os
import pickle
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()
chunk_size = int(os.environ.get('CHUNK_SIZE', 1000))
chunk_overlap = int(os.environ.get('CHUNK_OVERLAP', 100))
embeddingModel = os.environ.get('EMBEDDING_MODEL')

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "cached_embeddings.pkl")
SPLIT_DOCS_PATH = os.path.join(CACHE_DIR, "cached_split_docs.pkl")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")


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
    """Divide documentos em chunks e atribui metadados."""
    if os.path.exists(SPLIT_DOCS_PATH):
        print("Carregando documentos divididos do cache...")
        with open(SPLIT_DOCS_PATH, "rb") as f:
            split_docs = pickle.load(f)
        return split_docs

    print("Dividindo documentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []

    for item in raw_documents:
        texto_lei = item["texto_lei"]
        numero_lei = item["numero_lei"]  # Sabemos que não é vazio

        chunks = text_splitter.split_text(texto_lei)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={"numero_lei": numero_lei}
            )
            docs.append(doc)

    with open(SPLIT_DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"Total de pedaços gerados: {len(docs)}")
    return docs


def create_or_load_vectorstore(split_docs, embeddings):
    """Carrega FAISS se existir, senão cria e salva."""
    if os.path.exists(FAISS_INDEX_PATH):
        print("Carregando FAISS do disco...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS carregado com sucesso!")
    else:
        print("Criando base vetorial FAISS...")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print("Base vetorial FAISS criada e salva com sucesso!")
    return vector_store
