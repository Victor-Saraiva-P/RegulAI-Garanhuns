import json
from rouge_score import rouge_scorer
from config import *  
from db import load_or_fetch_documents  
from embeddings import create_embeddings, split_documents, create_or_load_vectorstore
from llm import init_llm, rag_search

# Inicializa o sistema RAG
raw_documents = load_or_fetch_documents()
embeddings = create_embeddings()
split_docs = split_documents(raw_documents)
vector_store = create_or_load_vectorstore(split_docs, embeddings)
llm = init_llm()

# Perguntas de teste e respostas esperadas
test_cases = [
    {
      "pergunta": "Quando é o dia do garçom?", 
      "resposta_esperada": "Segundo a Lei Nº 5.305/2024, o DIA MUNICIPAL DO GARÇOM é comemorado anualmente na segunda segunda-feira do mês de agosto."
    },
]

# Inicializa o ROUGE Scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

results = []

# Testa cada pergunta
for case in test_cases:
    pergunta = case["pergunta"]
    resposta_esperada = case["resposta_esperada"]
    
    # Obtém a resposta do RAG
    resposta_gerada = rag_search(pergunta, vector_store, llm).content

    # Calcula ROUGE Score
    scores = scorer.score(resposta_esperada, resposta_gerada)

    # Salva os resultados
    results.append({
        "pergunta": pergunta,
        "resposta_esperada": resposta_esperada,
        "resposta_gerada": resposta_gerada,
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    })

# Salva os resultados em um JSON para análise posterior
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Exibe os resultados no console
for r in results:
    print(f"\n🔹 Pergunta: {r['pergunta']}")
    print(f"✅ Esperada: {r['resposta_esperada']}")
    print(f"🤖 Gerada: {r['resposta_gerada']}")
    print(f"📊 ROUGE-1: {r['rouge1']:.4f}, ROUGE-L: {r['rougeL']:.4f}")
