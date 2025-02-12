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
    {
        "pergunta": "O que estabelece a Lei Nº 5.272/2024?",
        "resposta_esperada": "A Lei Nº 5.272/2024 institui a Política de Conscientização e Incentivo à Doação de Sangue, Órgãos, Tecidos e Leite Materno, chamada Promoção 3D."
    },
    {
        "pergunta": "Qual o critério de prioridade para matrícula escolar?",
        "resposta_esperada": "A prioridade de matrícula na rede municipal de ensino de Garanhuns é garantida em diversas situações. Entre elas: \n\n- Para crianças e adolescentes cujos pais ou responsáveis sejam idosos ou pessoas com deficiência (Art. 1º da Lei Nº 5.209/2024);\n- Para irmãos, garantindo preferência de matrícula na unidade escolar mais próxima de sua residência (Art. 2º);\n- Para estudantes que possuam os mesmos representantes legais, incluindo casos de guarda, tutela ou processo de adoção em andamento;\n- Para estudantes da zona rural matriculados na rede municipal de ensino (Art. 6º);\n- Para estudantes da zona urbana quando não houver vaga na escola mais próxima de sua residência, considerando critérios de setorização e geolocalização (Art. 6º)."
    },
    {
        "pergunta": "Quais são os critérios para vacinação domiciliar de idosos em Garanhuns?",
        "resposta_esperada": "De acordo com a Lei Nº 4211/2015, os critérios para vacinação domiciliar de idosos em Garanhuns são:\n\n- O idoso deve ter idade igual ou superior a 60 anos (Art. 2º);\n- A vacinação deve ser realizada em seu domicílio sempre que houver campanhas de imunização em andamento (§ 1º do Art. 1º);\n- A vacinação pode ser aplicada em residências ou em entidades que prestam assistência a idosos, como asilos e casas de repouso (Art. 2º)."
    },
    {
        "pergunta": "Existe alguma lei que trata da construção de sumidouros em Garanhuns?",
        "resposta_esperada": "Sim, a Lei Nº 4.546/2019 torna obrigatória a construção de sumidouros, incluindo caixa de inspeção e fossa séptica, para casas e prédios residenciais no município de Garanhuns, garantindo que os parâmetros técnicos sejam seguidos para o adequado descarte de dejetos."
    },
    {
        "pergunta": "O que estabelece a Lei Nº 5.099/2023 sobre saúde bucal nas escolas?",
        "resposta_esperada": "A Lei Nº 5.099/2023 institui a 'Semana Municipal da Saúde Bucal em Creches e Pré-Escolas da Rede Municipal de Ensino', que acontece anualmente na semana do Dia Nacional da Saúde Bucal, em 25 de outubro."
    },
    {
        "pergunta": "Quando é celebrado o Dia Municipal do Reciclador em Garanhuns?",
        "resposta_esperada": "De acordo com a Lei Nº 5.132/2023, o Dia Municipal do Reciclador é comemorado anualmente no dia 17 de maio."
    },
    {
        "pergunta": "Quando é comemorado o Dia dos Profissionais de Enfermagem e Técnico de Enfermagem em Garanhuns?",
        "resposta_esperada": "A Lei Nº 5.114/2023 estabelece que essa data é comemorada anualmente no dia 20 de maio."
    },
    {
        "pergunta": "Qual o valor da operação de crédito autorizada pela Lei Nº 5.010/2023?",
        "resposta_esperada": "A Lei Nº 5.010/2023 autoriza o Poder Executivo a contratar operação de crédito com a Caixa Econômica Federal, com ou sem garantia da União, no valor de até R$ 100.000.000,00 (cem milhões de reais)."
    }
]

# Inicializa o ROUGE Scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

results = []

# Testa cada pergunta
for case in test_cases:
    pergunta = case["pergunta"]
    resposta_esperada = case["resposta_esperada"]
    
    # Obtém a resposta do RAG
    resposta_gerada = rag_search(pergunta, vector_store, llm)

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
