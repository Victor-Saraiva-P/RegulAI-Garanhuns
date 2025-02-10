# config.py
import os
from dotenv import load_dotenv
from pyngrok import ngrok

# Carrega as variáveis do arquivo .env
load_dotenv()

# Configurações do ngrok
NGROK_KEY = os.environ.get("NGROK_KEY")
if NGROK_KEY:
    ngrok.set_auth_token(NGROK_KEY)

# Configurações do LangSmith (ou desabilite se não for necessário)
os.environ.setdefault("LANGSMITH_TRACING", "true")
# Se não precisar do tracing, defina para false:
# os.environ["LANGSMITH_TRACING"] = "false"

# Outras configurações podem ser adicionadas aqui
