import os
import openai
import logging
import time
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache

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
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Fast & cost-effective
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # Better for long document matching, Higher accuracy

db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Track FAISS modification time
last_loaded_faiss_time = os.path.getmtime(FAISS_INDEX_PATH + "/index.faiss")

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")


# ✅ Retry Mechanism for OpenAI API Calls
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


# ✅ Query Preprocessing to Improve Retrieval
def preprocess_query(user_query):
    """Enhance the query before retrieving documents (e.g., clarify vague questions)."""
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


# ✅ Reload FAISS Index if It Was Updated
def reload_faiss_if_needed():
    """Check if the FAISS index has been updated and reload it dynamically."""
    global db, last_loaded_faiss_time

    latest_faiss_time = os.path.getmtime(FAISS_INDEX_PATH + "/index.faiss")
    if latest_faiss_time > last_loaded_faiss_time:
        logging.info("🔄 Detected FAISS update, reloading index...")
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
        last_loaded_faiss_time = latest_faiss_time  # Update the last load timestamp


# ✅ Cache Responses to Reduce API Calls
@lru_cache(maxsize=50)  # Caches last 50 responses
def cached_response(user_query):
    return get_response(user_query)


def get_response(user_query):
    """Retrieve relevant documents and generate a response with sources using GPT-4."""
    global db  # Ensure we are using the latest FAISS index

    # ✅ Check if FAISS needs reloading
    reload_faiss_if_needed()

    # ✅ Preprocess query to improve retrieval
    user_query = preprocess_query(user_query)
    logging.info(f"User query after preprocessing: {user_query}")

    # ✅ Retrieve relevant document chunks from FAISS
    docs = db.similarity_search(user_query, k=8)  # Increased `k` for better retrieval

    if not docs:
        logging.warning("No relevant documents found.")
        return "Sorry, I couldn't find relevant information in the HR documents."

    # Extract text and source filenames
    retrieved_texts = []
    sources = set()

    for doc in docs:
        retrieved_texts.append(doc.page_content)
        
        # ✅ Debugging: Print document metadata
        logging.info(f"Retrieved document metadata: {doc.metadata}")

        if "source" in doc.metadata:
            sources.add(doc.metadata["source"])  # Store document names

    retrieved_text = "\n\n".join(retrieved_texts)

    logging.info(f"Retrieved document sections: {retrieved_text}")

    # ✅ Format query for GPT-4
    prompt = f"""
    You are an HR assistant. Answer the employee's question using only the provided HR documents.
    Ensure the answer is **well-structured, detailed, and clearly formatted**.

    ### HR Policy Information:
    {retrieved_text}

    ### Question:
    {user_query}

    ### Answer:
    """

    # ✅ Use robust OpenAI call with retry mechanism
    answer = robust_openai_call(prompt)

    # ✅ Add source references at the end of the response
    if sources:
        formatted_sources = ", ".join(sources)
        answer += f"\n\n(Source: {formatted_sources})"

    return answer


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
    response = cached_response(user_query)  # ✅ Uses caching

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
