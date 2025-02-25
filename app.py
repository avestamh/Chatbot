## Author: Sadra Avestan Feb 2025
import os
import openai
import logging
import time
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache

import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# FAISS Index Path
FAISS_INDEX_PATH = "./dbs/docs/faiss_index"

# Load FAISS database with embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Track FAISS modification time
last_loaded_faiss_time = os.path.getmtime(FAISS_INDEX_PATH + "/index.faiss")

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# List of potentially biased terms
biased_terms = ["gender", "race", "religion", "disability", "ethnicity", "age", "sexual orientation"]

def detect_bias_in_response(response_text):
    """Detects biased language in the chatbot's response."""
    flagged_terms = [term for term in biased_terms if re.search(rf'\b{term}\b', response_text, re.IGNORECASE)]
    return flagged_terms

#  Retry Mechanism for OpenAI API Calls
def robust_openai_call(prompt):
    """Retries OpenAI API call up to 3 times before failing."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"OpenAI API failed (attempt {attempt+1}/3): {e}")
            time.sleep(2)  # Wait before retrying
    return "There was an error processing your request. Please try again later."


# Query Preprocessing to Improve Retrieval
def preprocess_query(user_query):
    """Enhance the query before retrieving documents."""
    rephrase_prompt = f"Rephrase the following query to be more specific for an HR document search:\n\nQuery: {user_query}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": rephrase_prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"Could not rephrase query: {e}")
        return user_query  # Fallback to original query


# Reranking Retrieved Documents
def rerank_documents(user_query, docs):
    """Uses GPT-4 to rerank retrieved document chunks based on relevance."""
    doc_texts = [doc.page_content for doc in docs]
    concatenated_docs = "\n\n".join([f"Document {i+1}: {text}" for i, text in enumerate(doc_texts)])

    rerank_prompt = f"""
    You are an intelligent assistant designed to evaluate document relevance.
    Given the following user query and document chunks, rank the documents from most to least relevant.

    ### User Query:
    {user_query}

    ### Documents:
    {concatenated_docs}

    ### Instructions:
    List the document numbers from most to least relevant, separated by commas.

    Example output: 2, 1, 3, 4
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": rerank_prompt}]
        )
        ranked_order = response.choices[0].message.content.strip()
        logging.info(f"Re-ranked document order: {ranked_order}")

        # Convert the ranking output into a list of document indices
        ranked_indices = [int(num.strip()) - 1 for num in ranked_order.split(",") if num.strip().isdigit()]
        reranked_docs = [docs[i] for i in ranked_indices if i < len(docs)]

        return reranked_docs

    except Exception as e:
        logging.error(f"Failed to rerank documents: {e}")
        return docs  # Fallback to original order if reranking fails


# Reload FAISS Index if It Was Updated
def reload_faiss_if_needed():
    """Check if the FAISS index has been updated and reload it dynamically."""
    global db, last_loaded_faiss_time

    latest_faiss_time = os.path.getmtime(FAISS_INDEX_PATH + "/index.faiss")
    if latest_faiss_time > last_loaded_faiss_time:
        logging.info("🔄 Detected FAISS update, reloading index...")
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        last_loaded_faiss_time = latest_faiss_time


# Cache Responses to Reduce API Calls
@lru_cache(maxsize=50)
def cached_response(user_query):
    return get_response(user_query)


# Main Logic to Generate a Response
def get_response(user_query):
    global db
    reload_faiss_if_needed()

    # Preprocess the query
    user_query = preprocess_query(user_query)
    logging.info(f"User query after preprocessing: {user_query}")

    # Retrieve document chunks
    docs = db.similarity_search(user_query, k=8)

    if not docs:
        logging.warning("No relevant documents found.")
        return "Sorry, I couldn't find relevant information in the HR documents."

    # Rerank documents using GPT-4
    docs = rerank_documents(user_query, docs)

    # Extract top 5 document chunks for the final answer
    retrieved_texts = []
    sources = set()

    for doc in docs[:5]:
        retrieved_texts.append(doc.page_content)
        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])

    retrieved_text = "\n\n".join(retrieved_texts)

    # Final answer prompt
    prompt = f"""
    You are an HR assistant. Answer the employee's question using only the provided HR documents.
    Ensure the answer is **well-structured, detailed, and clearly formatted**.

    ### HR Policy Information:
    {retrieved_text}

    ### Question:
    {user_query}

    ### Answer:
    """

    # Generate the answer using GPT-4
    answer = robust_openai_call(prompt)
    # Run bias detection on the generated answer
    flagged_terms = detect_bias_in_response(answer)
    if flagged_terms:
        logging.warning(f"Potential bias detected in response: {flagged_terms}")
        answer += "\n\n⚠️ **Notice:** This response may contain sensitive terms. Please verify with HR for clarity."


    # Add sources to the response
    if sources:
        formatted_sources = ", ".join(sources)
        answer += f"\n\n(Source: {formatted_sources})"

    return answer


# Flask API Endpoints
@app.route("/")
def home():
    """Serve the chatbot UI."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """API endpoint for querying the chatbot."""
    data = request.get_json()

    if "question" not in data:
        return jsonify({"error": "Please provide a 'question' field"}), 400

    user_query = data["question"]
    response = cached_response(user_query)

    return jsonify({"response": response})


# Run the App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
