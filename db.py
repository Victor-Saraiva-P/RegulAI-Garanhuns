# db.py
from pymongo import MongoClient
import os
import pickle

MONGO_URI = (
    f"mongodb+srv://victoralex07062005:{os.environ.get('DB_PASSWORD')}"
    "@regulai-garanhuns-clust.mlrpw.mongodb.net/?retryWrites=true&w=majority&appName=RegulAI-Garanhuns-Cluster"
)
DATABASE_NAME = "RegulAI_Garanhuns"
COLLECTION_NAME = "leis_municipais"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

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
    """
    Carrega documentos do cache ou busca no MongoDB.
    Agora retorna APENAS os que possuam 'numero_lei' não vazio.
    """
    if os.path.exists(DOCS_PATH):
        print("Carregando documentos do cache...")
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
    else:
        print("Carregando documentos do MongoDB pela primeira vez...")
        collection = get_mongo_collection()
        documents = []

        # Busca apenas campos texto_lei e numero_lei
        for doc in collection.find({}, {"_id": 0, "texto_lei": 1, "numero_lei": 1}):
            texto_lei = doc.get("texto_lei", "").strip()
            numero_lei = doc.get("numero_lei", "").strip()

            # Se não tiver numero_lei ou estiver vazio, pular
            if not numero_lei:
                continue

            # Se também quiser garantir que texto_lei não seja vazio
            if not texto_lei:
                continue

            documents.append({
                "texto_lei": texto_lei,
                "numero_lei": numero_lei
            })

        print(f"Total de documentos carregados (com numero_lei): {len(documents)}")

        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)

    return documents
