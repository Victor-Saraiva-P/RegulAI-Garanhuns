# embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def create_embeddings():
    print("Criando embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embeddings criados!")
    return embeddings

def split_documents(raw_documents):
    print("Dividindo documentos em pedaços menores...")
    docs = [Document(page_content=doc) for doc in raw_documents]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    print(f"Total de pedaços gerados: {len(split_docs)}")
    return split_docs

def create_vectorstore(split_docs, embeddings):
    print("Criando base vetorial com FAISS...")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    print("Base vetorial criada com sucesso!")
    return vector_store
