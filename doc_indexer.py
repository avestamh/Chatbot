import os
import logging
import faiss
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# ✅ Define chunking strategy (Larger Chunks with Overlap)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # ✅ Increased for better context
    chunk_overlap=300  # ✅ Ensures better continuity between chunks
)

# ✅ Directory containing HR documents
DATA_PATH = "./docs/"
FAISS_INDEX_PATH = "./dbs/docs/faiss_index"

# ✅ Load existing FAISS database (if available)
if os.path.exists(FAISS_INDEX_PATH):
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    logging.info("✅ Existing FAISS index loaded.")
else:
    # ✅ Initialize FAISS with an empty index that supports incremental updates
    dimension = len(embeddings.embed_query("test"))  # Get embedding dimension
    faiss_index = faiss.IndexFlatL2(dimension)
    db = FAISS(faiss_index, embeddings, InMemoryDocstore({}), {})
    logging.info("⚠️ No existing FAISS index found. Creating a new one.")

documents = []

# ✅ Scan the docs directory for all supported files
for filename in os.listdir(DATA_PATH):
    file_path = os.path.join(DATA_PATH, filename)

    if file_path.lower().endswith((".pdf", ".docx", ".pptx", ".xlsx")):
        logging.info(f"📄 Processing: {file_path}")

        try:
            # ✅ Load file based on format
            if file_path.endswith(".pdf"):
                loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="ocr_only")
            else:
                loader = UnstructuredFileLoader(file_path)

            docs = loader.load()

            # ✅ Apply chunking
            split_docs = text_splitter.split_documents(docs)

            # ✅ Add filename as metadata for reference tracking
            for doc in split_docs:
                doc.metadata["source"] = filename  

            documents.extend(split_docs)

        except Exception as e:
            logging.error(f"❌ Error processing {filename}: {e}")

# ✅ Update or Create FAISS index
if documents:
    if db.index.ntotal == 0:  # ✅ If FAISS index is empty, create a new one
        db = FAISS.from_documents(documents, embedding=embeddings)
    else:
        db.add_documents(documents)  # ✅ Properly appends new documents

    db.save_local(FAISS_INDEX_PATH)
    logging.info("✅ Indexing complete. FAISS vector store updated.")
else:
    logging.warning("⚠️ No valid documents found for indexing.")
