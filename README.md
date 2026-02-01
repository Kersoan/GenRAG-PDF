# GenRAG-PDF  
**Retrieval-Augmented Generation System for Multi-PDF Question Answering**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Overview
GenRAG-PDF is an AI-powered application that enables users to query and extract insights from **multiple PDF documents** using a **Retrieval-Augmented Generation (RAG)** pipeline.  
The system combines semantic search with large language models to generate accurate, context-aware responses.

---

## 🧠 Key Features
- Multi-PDF document ingestion
- Text chunking and semantic embeddings
- FAISS-based vector search
- Google Gemini-powered response generation
- Fallback text-matching when API quota is exceeded
- Clean Streamlit-based UI

---

## 🏗 Architecture
1. PDF text extraction using **PyPDF2**
2. Chunking with **RecursiveCharacterTextSplitter**
3. Embedding generation using **Google Gemini**
4. Vector storage using **FAISS**
5. Context-aware answer generation via **RAG**

---

## 🛠 Tech Stack
- **Language:** Python  
- **Framework:** Streamlit  
- **LLM:** Google Gemini  
- **Vector Store:** FAISS  
- **Libraries:** LangChain, PyPDF2, dotenv  

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/Kersoan/GenRAG-PDF.git
cd GenRAG-PDF
pip install -r requirements.txt
streamlit run app/app.py
