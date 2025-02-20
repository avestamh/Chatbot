import json
import os
import re
import torch
import logging
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# ✅ Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ ERROR: OpenAI API key is missing. Check your .env file.")

# ✅ Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Load FAISS index
FAISS_INDEX_PATH = "./dbs/docs/faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

# ✅ Load test questions and expected answers
with open("test_data.json", "r") as f:
    test_cases = json.load(f)

# ✅ Initialize ROUGE scorer & SBERT model
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# ✅ Function to normalize text for better ROUGE evaluation
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[-•]", "", text)  # Remove bullet points
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.strip()

# ✅ Function to compute semantic similarity (SBERT, now using GPU)
def compute_semantic_similarity(expected_answer, chatbot_response):
    """Use GPU if available to speed up SBERT similarity computation."""
    embedding1 = sbert_model.encode(expected_answer, convert_to_tensor=True).to(device)
    embedding2 = sbert_model.encode(chatbot_response, convert_to_tensor=True).to(device)

    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

# ✅ Function to re-rank retrieved results using GPT-4 (Now passing full context)
def rerank_results(query, full_retrieved_text):
    """Send full concatenated retrieved text to GPT-4 for ranking."""
    
    prompt = f"Based on the following HR documents, answer the query: '{query}'\n\n{full_retrieved_text}"
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content  # ✅ Return full-length text

# ✅ Evaluate chatbot performance
results = []

for test in test_cases:
    question = test["question"]
    expected_answer = test["expected_answer"]
    expected_sources = set(test["expected_sources"])

    # ✅ Retrieve documents from FAISS (Increase `k=8` for better context)
    retrieved_docs = db.similarity_search(question, k=8)  # Increased k

    retrieved_texts = []
    retrieved_sources = set()

    for doc in retrieved_docs:
        retrieved_texts.append(doc.page_content)
        if "source" in doc.metadata:
            retrieved_sources.add(doc.metadata["source"])

    # ✅ Concatenate all retrieved chunks into one full-length text
    full_retrieved_text = "\n\n".join(retrieved_texts)

    # ✅ Send full context to GPT-4 for better responses
    best_retrieved_text = rerank_results(question, full_retrieved_text)

    # ✅ Compute ROUGE score
    rouge_scores = scorer.score(normalize_text(expected_answer), normalize_text(best_retrieved_text))

    # ✅ Compute Semantic Similarity (SBERT on GPU)
    semantic_similarity = compute_semantic_similarity(expected_answer, best_retrieved_text)

    # ✅ Compute Recall@K
    recall_at_k = len(expected_sources.intersection(retrieved_sources)) / len(expected_sources) if expected_sources else 0.0

    # ✅ Store results (Now includes full answer & sources)
    results.append({
        "question": question,
        "rouge_1": rouge_scores["rouge1"].fmeasure,
        "rouge_2": rouge_scores["rouge2"].fmeasure,
        "semantic_similarity": semantic_similarity,
        "recall_at_k": recall_at_k,
        "retrieved_answer": best_retrieved_text,  # ✅ Now full-length
        "retrieved_sources": list(retrieved_sources)  # ✅ Includes sources
    })

# ✅ Print summary
for res in results:
    print(f"\n🔹 Question: {res['question']}")
    print(f" ROUGE-1: {res['rouge_1']:.4f}, ROUGE-2: {res['rouge_2']:.4f}")
    print(f" Semantic Similarity (SBERT): {res['semantic_similarity']:.4f}")
    print(f" Recall@K: {res['recall_at_k']:.2f}")
    print(f" Retrieved Sources: {res['retrieved_sources']}")
    print(f" Retrieved Answer:\n{res['retrieved_answer']}\n")
