The file includes the app.py to run the chatbot, python_indexer.py to indext the file, whatcher.py to watch for new document, delted of modified document to re-index the db file. It does it dynamically without the need to run python_indexte on the entire data base.

# 🏢  Chatbot with Dynamic RAG and FAISS Integration

A dynamic HR chatbot powered by **GPT-4** and **FAISS**, designed to assist employees with HR-related queries. The system supports real-time document updates, multiple file formats (PDF, DOCX, PPTX, XLSX), and scalable deployment on both on-premises and cloud platforms.

## 🚀 Features
- 🔍 **Dynamic Document Retrieval** with FAISS and real-time indexing.
-  **GPT-4 Response Generation** with document reranking for improved relevance.
-  **Role-Based Access Control** to ensure hierarchical access to HR documents.
-  **Real-Time Monitoring** for automatic document updates using a Watcher Service.
-  **Evaluation Metrics** including ROUGE, SBERT Semantic Similarity, and Recall@K.
- ⚖ **Responsible AI Compliance** with built-in bias detection and transparency features.

## ⚙️ Tech Stack
- **Python**, **Flask** for API and web interface
- **FAISS** for efficient vector similarity search
- **LangChain** for advanced document processing
- **OpenAI GPT-4** for generating accurate responses
- **Docker** for containerization and deployment
- **CI/CD Pipelines** for automated deployment
- **OCR Integration** for handling documents with embedded images

![spin_clear_button_new](https://github.com/user-attachments/assets/6017440b-06c3-4391-a49e-25048fdeba0b)

##  Usage
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/avestamh/chatbot.git

