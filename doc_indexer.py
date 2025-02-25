import os
import logging
import shutil
import faiss
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Define chunking strategy (Increased chunk size for better retrieval)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Larger chunks for better context
    chunk_overlap=400  # More overlap for continuity
)

#  Define directories
DATA_PATH = "./docs/"
FAISS_INDEX_PATH = "./dbs/docs/faiss_index"

# Clear existing FAISS index to avoid mixing old data
if os.path.exists(FAISS_INDEX_PATH):
    shutil.rmtree(FAISS_INDEX_PATH)
    logging.info("⚠️ Old FAISS index cleared before re-indexing.")

# Initialize a new FAISS index
dimension = len(embeddings.embed_query("test"))  # Get embedding dimension
faiss_index = faiss.IndexFlatL2(dimension)
db = FAISS(faiss_index, embeddings, InMemoryDocstore({}), {})
logging.info(" Initialized a new FAISS index.")

#  Prepare documents for indexing
documents = []

# Load and process documents
for filename in os.listdir(DATA_PATH):
    file_path = os.path.join(DATA_PATH, filename)

    if file_path.lower().endswith((".pdf", ".docx", ".pptx", ".xlsx")):
        logging.info(f"📄 Processing: {file_path}")

        try:
            # Load documents based on format
            if file_path.endswith(".pdf"):
                loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="ocr_only")
            else:
                loader = UnstructuredFileLoader(file_path)

            docs = loader.load()

            # Apply chunking to the documents
            split_docs = text_splitter.split_documents(docs)

            # Attach metadata for each chunk
            for doc in split_docs:
                doc.metadata["source"] = filename  # Add source for traceability

            documents.extend(split_docs)

        except Exception as e:
            logging.error(f"❌ Error processing {filename}: {e}")

#  Update or Create FAISS index
if documents:
    db = FAISS.from_documents(documents, embedding=embeddings)  # Fully rebuild index
    db.save_local(FAISS_INDEX_PATH)
    logging.info(" Indexing complete. FAISS vector store successfully updated.")
else:
    logging.warning("⚠️ No valid documents found for indexing.")
