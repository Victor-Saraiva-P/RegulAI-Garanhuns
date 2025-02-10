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

os.environ.setdefault("LANGSMITH_TRACING", "true")

