#Advanced LLM-Based Document Question Answering System

This project is an advanced, production-ready **Retrieval-Augmented Generation (RAG)** system that allows users to upload multiple documents (PDF, DOCX, TXT, etc.) and ask questions about their content.
It combines document embedding, vector search, and large language models (LLMs) such as OpenAI GPT or local models like Llama 3 or Mistral.

---

## 1. Overview

The system uses the following components:

1. **Document Loader** – Reads and splits documents into smaller text chunks.
2. **Vector Store** – Creates and stores embeddings for each chunk using Chroma or FAISS.
3. **Retriever** – Finds the most relevant document chunks for each user query.
4. **LLM Engine** – Uses OpenAI or a local model (e.g., Llama 3) to generate answers.
5. **Conversational Memory** – Maintains chat context for follow-up questions.
6. **Web Interface** – A Streamlit application that enables uploading documents and interacting with the QA engine.

---

## 2. Features

* Supports **PDF**, **DOCX**, and **TXT** document types
* Multi-document ingestion and processing
* **Persistent** vector database (Chroma)
* **Conversational memory** for context-aware Q&A
* Switch between **OpenAI** and **local models** (via Llama-cpp or Ollama)
* Easy-to-use **Streamlit web UI**
* Modular, extensible codebase

---

## 3. Project Structure

```
advanced_doc_qa/
│
├── app.py                     # Streamlit frontend
├── config.py                  # Configuration (API keys, model choice, paths)
├── core/
│   ├── document_loader.py     # Document ingestion and splitting
│   ├── vector_store.py        # Embedding creation and vector database logic
│   ├── qa_engine.py           # Conversational QA chain with memory
│   └── utils.py               # Utility functions (optional)
├── data/
│   └── sample_docs/           # Example or uploaded documents
├── db/                        # Persistent Chroma vector database
├── requirements.txt
└── README.md
```

---

## 4. Installation

### Prerequisites

* Python 3.9 or higher
* pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/advanced_doc_qa.git
cd advanced_doc_qa
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

Open `config.py` and set the following:

```python
USE_OPENAI = True  # Set to False to use a local model
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
CHROMA_DB_DIR = "db/chroma_store"
EMBEDDING_MODEL = "text-embedding-3-large"
```

If you want to use a **local model**, install and configure **llama-cpp** or **Ollama** and update `USE_OPENAI = False`.

---

## 5. Running the Application

Run the Streamlit interface:

```bash
streamlit run app.py
```

Once the web interface opens:

1. Upload one or more documents (PDF, TXT, or DOCX).
2. Wait for the system to process and index the content.
3. Type a question and click **Ask**.
4. View the generated answer and the retrieved context snippet.

---

## 6. Example Workflow

1. Upload `contract.pdf` and `company_policy.docx`.
2. Ask:

   ```
   What are the key terms of the contract?
   ```
3. Follow up:

   ```
   Does the contract mention early termination conditions?
   ```
4. The system remembers context and provides accurate, document-based answers.

---

## 7. Configuration Options

| Setting                         | Description                            | Default                  |
| ------------------------------- | -------------------------------------- | ------------------------ |
| `USE_OPENAI`                    | Toggle between OpenAI and local model  | `True`                   |
| `CHROMA_DB_DIR`                 | Directory for vector store persistence | `db/chroma_store`        |
| `EMBEDDING_MODEL`               | Embedding model for vectorization      | `text-embedding-3-large` |
| `model_path` (for local models) | Path to `.gguf` model file             | `models/llama-3.gguf`    |

---

## 8. Extending the System

This framework can be extended to support:

* Multiple retrieval methods (RAG fusion)
* Vector database alternatives (FAISS, Pinecone, Weaviate)
* Cloud deployment (AWS, Azure, Hugging Face Spaces)
* Authentication and user sessions
* Enhanced frontend (React or Next.js)
* Summarization and document search features

---

## 9. Troubleshooting

**Issue:** Slow response time
**Solution:** Use smaller embedding models or reduce chunk size.

**Issue:** OpenAI API errors
**Solution:** Verify your API key and network connection.

**Issue:** Local model not loading
**Solution:** Ensure `llama-3.gguf` or equivalent model file exists and paths are correctly set in `config.py`.

---

## 10. License

This project is released under the MIT License.
You may use, modify, and distribute it freely with attribution.

---

## 11. Acknowledgments

This project leverages the following open-source technologies:

* [LangChain](https://www.langchain.com)
* [Chroma](https://www.trychroma.com)
* [Streamlit](https://streamlit.io)
* [OpenAI API](https://platform.openai.com)
* [Llama-cpp](https://github.com/ggerganov/llama.cpp)
