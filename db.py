# db.py
from pymongo import MongoClient
import os
import streamlit as st
import logging

MONGO_URI = (
    f"mongodb+srv://victoralex07062005:{os.environ.get('DB_PASSWORD')}"
    "@regulai-garanhuns-clust.mlrpw.mongodb.net/?retryWrites=true&w=majority&appName=RegulAI-Garanhuns-Cluster"
)
DATABASE_NAME = "RegulAI_Garanhuns"
COLLECTION_NAME = "leis_municipais"


logger = logging.getLogger(__name__)

def get_mongo_collection():
    logger.info("Conectando ao MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    logger.info("Conexão com MongoDB estabelecida!")
    return collection


def load_documents(collection):
    documents = []
    logger.info("Carregando documentos do MongoDB...")
    for doc in collection.find({}, {"_id": 0, "texto_lei": 1}):
        if "texto_lei" in doc:
            documents.append(doc["texto_lei"])
    logger.info(f"Total de documentos carregados: {len(documents)}")
    return documents
