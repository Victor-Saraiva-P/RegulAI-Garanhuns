import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Criar diretórios se não existirem
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Caminhos para cache
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "cached_embeddings.pkl")
SPLIT_DOCS_PATH = os.path.join(CACHE_DIR, "cached_split_docs.pkl")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")  # FAISS cria múltiplos arquivos, melhor manter na pasta.

def create_embeddings():
    """Cria ou carrega os embeddings."""
    if os.path.exists(EMBEDDINGS_PATH):
        print("Carregando embeddings do cache...")
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("Gerando embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)
        print("Embeddings criados e salvos!")

    return embeddings

def split_documents(raw_documents):
    """Divide documentos ou carrega do cache."""
    if os.path.exists(SPLIT_DOCS_PATH):
        print("Carregando documentos divididos do cache...")
        with open(SPLIT_DOCS_PATH, "rb") as f:
            split_docs = pickle.load(f)
    else:
        print("Dividindo documentos...")
        docs = [Document(page_content=doc) for doc in raw_documents]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        # Salvar para não precisar dividir sempre
        with open(SPLIT_DOCS_PATH, "wb") as f:
            pickle.dump(split_docs, f)
        print(f"Total de pedaços gerados: {len(split_docs)}")

    return split_docs

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
