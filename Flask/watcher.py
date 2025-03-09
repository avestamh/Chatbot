# Author: Sadra Avestan Feb 2025
"""
Automates FAISS indexing using a File Watcher (real-time detection).
Instead of re-indexing all documents, it updates FAISS only for new, modified, or deleted files.
"""

import time
import os
import logging
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from plyer import notification
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Track recently processed files to prevent redundant indexing
recently_processed_files = {}

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory to watch
WATCH_FOLDER = "./docs/"
FAISS_INDEX_PATH = "./dbs/docs/faiss_index"

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Function to send desktop notifications
def send_desktop_notification(file_path, action="Indexed"):
    try:
        notification.notify(
            title="HR Document Update",
            message=f"Document {action}: {file_path}",
            timeout=5
        )
    except Exception as e:
        logging.warning(f"⚠️ Notification error: {e}")

# Helper function to generate a file hash (to track modifications)
def get_file_hash(file_path):
    """Returns a hash of the file's content to detect modifications."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Load existing FAISS database (if available)
if os.path.exists(FAISS_INDEX_PATH):
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)  # ✅ FIXED
else:
    db = None  # Initialize as empty

# File event handler
class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Triggered when a new document is added."""
        if event.is_directory or not event.src_path.lower().endswith((".pdf", ".docx", ".pptx", ".xlsx")):
            return

        # Check if the file is already indexed
        new_hash = get_file_hash(event.src_path)
        if event.src_path in recently_processed_files and recently_processed_files[event.src_path] == new_hash:
            logging.info(f"⚠️ File already indexed: {event.src_path} (Skipping)")
            return  

        logging.info(f"📢 New document added: {event.src_path}")
        send_desktop_notification(event.src_path, action="Added")

        # Index the new document
        self.index_new_document(event.src_path)

        # Store the new file hash
        recently_processed_files[event.src_path] = new_hash


    def on_modified(self, event):
        """Triggered when an existing document is modified."""
        if event.is_directory or not event.src_path.lower().endswith((".pdf", ".docx", ".pptx", ".xlsx")):
            return

        # Check if file content actually changed
        new_hash = get_file_hash(event.src_path)
        if event.src_path in recently_processed_files and recently_processed_files[event.src_path] == new_hash:
            logging.info(f"⚠️ Ignoring modification event (No content change): {event.src_path}")
            return  

        logging.info(f"🔄 Document modified: {event.src_path}")
        send_desktop_notification(event.src_path, action="Modified")

        # Step 1: Remove old version from FAISS
        self.remove_document(event.src_path)

        # Step 2: Add the updated document back into FAISS
        self.index_new_document(event.src_path)

        # Step 3: Update the stored hash
        recently_processed_files[event.src_path] = new_hash


    def on_deleted(self, event):
        """Triggered when a document is deleted."""
        if event.is_directory or not event.src_path.lower().endswith((".pdf", ".docx", ".pptx", ".xlsx")):
            return
        
        logging.warning(f"🗑️ Document deleted: {event.src_path}")
        send_desktop_notification(event.src_path, action="Deleted")

        # Remove the document from FAISS
        self.remove_document(event.src_path)

    def index_new_document(self, file_path):
        """Indexes a new document and appends it to FAISS."""
        global db

        logging.info(f"🔄 Indexing new document: {file_path}")

        try:
            if file_path.endswith(".pdf"):
                loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="ocr_only")
            else:
                loader = UnstructuredFileLoader(file_path)
                
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)

            if db is None:
                db = FAISS.from_documents(docs, embedding=embeddings)
            else:
                db.add_documents(docs)  # Use incremental indexing

            db.save_local(FAISS_INDEX_PATH)
            logging.info(f" Successfully indexed: {file_path}")

        except Exception as e:
            logging.error(f"❌ Error indexing {file_path}: {e}")

    def remove_document(self, file_path):
        """Removes a document’s embedding from FAISS without full reindexing."""
        global db

        if db is None:
            logging.warning("⚠️ No FAISS index found.")
            return  

        try:
            file_name = os.path.basename(file_path)
            logging.info(f"🗑️ Removing document from index: {file_name}")

            # Remove the document from FAISS without full reindexing
            new_docs = [
                doc for doc in db.similarity_search(" ", k=len(db.index_to_docstore_id))  
                if "source" in doc.metadata and doc.metadata["source"] != file_name
            ]

            if not new_docs:
                logging.info("⚠️ No documents remain. Resetting FAISS index.")
                db = None  # Reset FAISS if all documents are removed
            else:
                db = FAISS.from_documents(new_docs, embedding=embeddings)  # Rebuild FAISS

            db.save_local(FAISS_INDEX_PATH)
            logging.info(f" Successfully removed {file_name} from index.")

        except Exception as e:
            logging.error(f"❌ Error removing {file_path} from index: {e}")


# ✅ Start watching
if __name__ == "__main__":
    event_handler = DocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    
    logging.info(f"👀 Watching for HR document changes in: {WATCH_FOLDER}")
    observer.start()

    try:
        while True:
            time.sleep(5)  # Prevents CPU overload
    except KeyboardInterrupt:
        observer.stop()
        logging.info("❌ Stopped watching for document changes.")

    observer.join()
