from pymongo import MongoClient
import os
import pickle  # Para salvar e carregar documentos localmente

MONGO_URI = (
    f"mongodb+srv://victoralex07062005:{os.environ.get('DB_PASSWORD')}"
    "@regulai-garanhuns-clust.mlrpw.mongodb.net/?retryWrites=true&w=majority&appName=RegulAI-Garanhuns-Cluster"
)
DATABASE_NAME = "RegulAI_Garanhuns"
COLLECTION_NAME = "leis_municipais"

# Criar diretório para armazenar os dados processados
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Caminho do cache dos documentos processados
DOCS_PATH = os.path.join(DATA_DIR, "processed_documents.pkl")

def get_mongo_collection():
    """Estabelece conexão com o MongoDB"""
    print("Conectando ao MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    print("Conexão com MongoDB estabelecida!")
    return collection

def load_or_fetch_documents():
    """Carrega documentos do cache, se disponíveis. Caso contrário, busca no MongoDB."""
    if os.path.exists(DOCS_PATH):
        print("Carregando documentos do cache...")
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
    else:
        print("Carregando documentos do MongoDB pela primeira vez...")
        collection = get_mongo_collection()
        documents = []
        
        for doc in collection.find({}, {"_id": 0, "texto_lei": 1}):
            if "texto_lei" in doc:
                documents.append(doc["texto_lei"])

        print(f"Total de documentos carregados: {len(documents)}")

        # Salva os documentos no cache
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)

    return documents
