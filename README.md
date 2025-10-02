# **GenRAG-PDF (Chat with multiple PDF files using R A G)**

GenRAG-PDF is an AI-powered system that enables interactive querying of multiple PDF documents. Using a Retrieval-Augmented Generation (RAG) pipeline, it combines semantic search with generative AI to provide context-aware and accurate answers. The app extracts text from PDFs, splits it into meaningful chunks, creates embeddings (Google Gemini or fallback), and delivers answers via a Streamlit-based web interface.

# **How It Works:**
1.Upload one or multiple PDF documents.

2.System processes documents to extract text, create embeddings, and build a searchable index.

3.Users ask questions in the interface, and the system retrieves relevant content and generates context-aware answers.

4.Automatically switches to fallback embedding or simple text matching when API limits are reached.

