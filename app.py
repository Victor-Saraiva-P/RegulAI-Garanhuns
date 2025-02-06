import os
import streamlit as st
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pyngrok import ngrok
from dotenv import load_dotenv

# Carregar variáveis do arquivo .env
load_dotenv()

# Se necessário, você pode solicitar a entrada via getpass ou input, mas é melhor deixar no .env para execução local
# Exemplo:
# if not os.environ.get("HF_TOKEN"):
#     os.environ["HF_TOKEN"] = input("Digite sua API key do Hugging Face: ")

# Configuração do ngrok
ngrok_key = os.environ.get("NGROK_KEY")
if ngrok_key:
    ngrok.set_auth_token(ngrok_key)

# Conexão com MongoDB
db_password = os.environ.get("DB_PASSWORD")
MONGO_URI = (
    f"mongodb+srv://victoralex07062005:{db_password}@regulai-garanhuns-clust.mlrpw.mongodb.net/"
    "?retryWrites=true&w=majority&appName=RegulAI-Garanhuns-Cluster"
)
DATABASE_NAME = "RegulAI_Garanhuns"
COLLECTION_NAME = "leis_municipais"

# Cria o túnel ngrok (opcional, se desejar acesso externo)
if ngrok_key:
    public_url = ngrok.connect("http://localhost:8501", "http")
    st.write(f"Túnel do Ngrok criado: {public_url}")

# Conectar ao MongoDB
st.write("Conectando ao MongoDB...")
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]
st.write("Conexão com MongoDB estabelecida!")

# Verifica a existência de documentos
test_doc = collection.find_one()
if not test_doc:
    st.error("Nenhum documento encontrado no MongoDB! Verifique a conexão e os dados.")
    st.stop()
st.write("Exemplo de documento encontrado:", test_doc)

# Função para carregar os documentos do MongoDB (sem metadados)
def load_documents():
    documents = []
    st.write("Carregando documentos do MongoDB...")
    for doc in collection.find({}, {"_id": 0, "texto_lei": 1}):
        if "texto_lei" in doc:
            content = doc["texto_lei"]
            documents.append(Document(page_content=content))
    st.write(f"Total de documentos carregados: {len(documents)}")
    return documents

docs = load_documents()
if not docs:
    st.error("Nenhum documento válido encontrado. Encerrando execução.")
    st.stop()
st.write(f"Carregados {len(docs)} documentos do MongoDB")

# Criando embeddings e FAISS
st.write("Criando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
st.write("Embeddings criados!")

# Dividindo os documentos em pedaços menores para indexação
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
st.write("Dividindo documentos em pedaços menores...")
split_docs = text_splitter.split_documents(docs)
st.write(f"Total de pedaços gerados: {len(split_docs)}")
if not split_docs:
    st.error("Nenhum pedaço de documento encontrado para FAISS. Verifique a extração de dados.")
    st.stop()

# Criando a base vetorial com FAISS
st.write("Criando base vetorial com FAISS...")
vector_store = FAISS.from_documents(split_docs, embeddings)
st.write("Base vetorial criada com sucesso!")

# Configurando o LLM (Groq)
st.write("Configurando LLM Groq...")
llm = ChatGroq(model="llama3-8b-8192")
st.write("LLM configurado!")

# Função para normalizar número da lei (se necessário)
def normalize_law_number(law_number: str) -> str:
    law_number = law_number.strip()
    if '.' in law_number:
        return law_number
    parts = law_number.split('/')
    if len(parts) != 2:
        return law_number
    num_part, year_part = parts
    if len(num_part) > 1:
        normalized = num_part[0] + '.' + num_part[1:] + '/' + year_part
        return normalized
    return law_number

# Função de busca RAG somente via FAISS
def rag_search(query: str) -> str:
    st.write(f"\nRealizando busca para: {query}")

    # Busca vetorial via FAISS
    relevant_docs = vector_store.similarity_search(query, k=5)
    st.write(f"Documentos relevantes encontrados: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs):
        st.write(f"Documento {i+1} - Início do texto:\n{doc.page_content[:200]}...\n")

    # Concatena trechos para formar o contexto
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Baseando-se no texto das leis a seguir, responda com precisão:\n\n{context}\n\nPergunta: {query}\nResposta:"
    st.write("Enviando prompt para LLM...")
    
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        st.error(f"Erro ao invocar LLM: {e}")
        response = "Desculpe, ocorreu um erro ao gerar a resposta."
    
    st.write("Resposta gerada!")
    return response

# Interface do Chatbot com Streamlit
st.title("📜 RegulAI - Chatbot de Leis Municipais 🏛️")
st.write("Pergunte sobre leis municipais e receba respostas baseadas nos textos legais!")

user_input = st.text_input("Digite sua pergunta:")
if user_input:
    with st.spinner("Buscando resposta..."):
        response = rag_search(user_input)
        st.text_area("Resposta:", response, height=200)
