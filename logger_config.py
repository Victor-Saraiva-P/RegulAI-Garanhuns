import logging

# Configuração básica do logging
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG para mais detalhes
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
