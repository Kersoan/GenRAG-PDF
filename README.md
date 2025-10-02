# **GenRAG-PDF (Chat with multiple PDF files using R A G)**
###
-GenRAG-PDF is an advanced AI-powered application designed to enable users to interactively query and retrieve precise information from multiple PDF documents. Leveraging a Retrieval-Augmented Generation (RAG) ------pipeline, the system combines semantic search with generative AI to provide context-aware and highly accurate answers.
-PDF Text Extraction: Utilizes PyPDF2 to efficiently extract text from PDFs, including multi-page documents.
-Text Chunking: Employs LangChain’s RecursiveCharacterTextSplitter to divide large documents into meaningful chunks for better retrieval.
-Embedding & Vector Store:
  *Primary embeddings via Google Gemini API for semantic understanding.
  *Automatic fallback to Sentence-Transformers or simple text-based indexing in case of API quota limitations.
-Conversational AI: Integrates LangChain ChatGoogleGenerativeAI with a custom prompt template to generate detailed, contextually accurate responses.
-Interactive Web Interface: Built with Streamlit for a clean, intuitive, and user-friendly experience.
###
# **How It Works:**
1.Upload one or multiple PDF documents.
2.System processes documents to extract text, create embeddings, and build a searchable index.
3.Users ask questions in the interface, and the system retrieves relevant content and generates context-aware answers.
4.Automatically switches to fallback embedding or simple text matching when API limits are reached.

