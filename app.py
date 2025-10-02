import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain.vectorstores import FAISS
    except ImportError:
        # Custom FAISS implementation using raw faiss library
        import faiss
        from langchain.vectorstores.base import VectorStore
        from typing import List, Any, Optional
        import numpy as np
        
        class FAISS(VectorStore):
            def __init__(self, embedding_function, index=None, docstore=None, index_to_docstore_id=None):
                self.embedding_function = embedding_function
                self.index = index
                self.docstore = docstore or {}
                self.index_to_docstore_id = index_to_docstore_id or {}
            
            @classmethod
            def from_texts(cls, texts: List[str], embedding=None, **kwargs):
                try:
                    embeddings = embedding.embed_documents(texts)
                    dimension = len(embeddings[0])
                    index = faiss.IndexFlatL2(dimension)
                    index.add(np.array(embeddings).astype('float32'))
                    
                    docstore = {str(i): text for i, text in enumerate(texts)}
                    index_to_docstore_id = {i: str(i) for i in range(len(texts))}
                    
                    return cls(embedding, index, docstore, index_to_docstore_id)
                except Exception as e:
                    if "quota" in str(e).lower() or "429" in str(e):
                        # Re-raise the quota error to be handled by the calling function
                        raise e
                    else:
                        raise e
            
            def save_local(self, folder_path: str):
                import os
                os.makedirs(folder_path, exist_ok=True)
                faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
                import pickle
                with open(os.path.join(folder_path, "docstore.pkl"), "wb") as f:
                    pickle.dump(self.docstore, f)
                with open(os.path.join(folder_path, "index_to_docstore_id.pkl"), "wb") as f:
                    pickle.dump(self.index_to_docstore_id, f)
            
            @classmethod
            def load_local(cls, folder_path: str, embeddings):
                import os
                import pickle
                
                # Check if the index files exist
                index_file = os.path.join(folder_path, "index.faiss")
                docstore_file = os.path.join(folder_path, "docstore.pkl")
                index_to_docstore_file = os.path.join(folder_path, "index_to_docstore_id.pkl")
                
                if not os.path.exists(index_file):
                    raise FileNotFoundError(f"FAISS index not found at {folder_path}. Please upload and process PDF files first!")
                
                index = faiss.read_index(index_file)
                with open(docstore_file, "rb") as f:
                    docstore = pickle.load(f)
                with open(index_to_docstore_file, "rb") as f:
                    index_to_docstore_id = pickle.load(f)
                return cls(embeddings, index, docstore, index_to_docstore_id)
            
            def similarity_search(self, query: str, k: int = 4):
                try:
                    query_embedding = self.embedding_function.embed_query(query)
                    scores, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
                    
                    docs = []
                    for idx in indices[0]:
                        if idx in self.index_to_docstore_id:
                            doc_id = self.index_to_docstore_id[idx]
                            if doc_id in self.docstore:
                                docs.append(self.docstore[doc_id])
                    return docs
                except Exception as e:
                    if "quota" in str(e).lower() or "429" in str(e):
                        # Re-raise the quota error to be handled by the calling function
                        raise e
                    else:
                        raise e
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    try:
        # Try Google Gemini first
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("✅ **Using Google Gemini embeddings!** Processing completed.")
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            st.warning("⚠️ **Google Gemini quota exceeded. Using alternative embeddings...**")
            try:
                # Skip PyTorch entirely and use simple text matching directly
                st.warning("⚠️ **Using simple text matching to avoid PyTorch issues...**")
                
                import re
                import os
                import pickle
                
                # Simple text similarity approach (no PyTorch needed)
                def create_text_index(texts):
                    """Create a simple text-based index"""
                    index_data = {
                        'texts': texts,
                        'word_to_docs': {},
                        'doc_word_counts': []
                    }
                    
                    for doc_id, text in enumerate(texts):
                        words = set(re.findall(r'\w+', text.lower()))
                        index_data['doc_word_counts'].append(len(words))
                        
                        for word in words:
                            if word not in index_data['word_to_docs']:
                                index_data['word_to_docs'][word] = []
                            index_data['word_to_docs'][word].append(doc_id)
                    
                    return index_data
                
                # Create simple index
                text_index = create_text_index(text_chunks)
                
                # Save simple index
                os.makedirs("faiss_index", exist_ok=True)
                with open(os.path.join("faiss_index", "simple_index.pkl"), "wb") as f:
                    pickle.dump(text_index, f)
                with open(os.path.join("faiss_index", "embedding_type.txt"), "w") as f:
                    f.write("simple-text")
                
                st.success("✅ **Using simple text matching!** Processing completed.")
                
            except Exception as fallback_error:
                st.error(f"❌ Fallback failed: {str(fallback_error)}")
                raise
        else:
            st.error(f"❌ Error creating vector store: {str(e)}")
            raise


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    try:
        # Check if FAISS index exists
        if not os.path.exists("faiss_index"):
            st.error("❌ No PDF files processed yet! Please upload PDF files and click 'Submit & Process' first.")
            return
        
        # Check which embedding type was used
        embedding_type = "google_gemini"  # default
        if os.path.exists(os.path.join("faiss_index", "embedding_type.txt")):
            with open(os.path.join("faiss_index", "embedding_type.txt"), "r") as f:
                embedding_type = f.read().strip()
        
        if embedding_type == "simple-text":
            # Use simple text matching for search
            st.info("🔍 **Searching with simple text matching...**")
            import pickle
            import re
            
            # Load simple index
            with open(os.path.join("faiss_index", "simple_index.pkl"), "rb") as f:
                text_index = pickle.load(f)
            
            # Simple text similarity search
            query_words = set(re.findall(r'\w+', user_question.lower()))
            if len(query_words) == 0:
                st.write("Please enter a valid question.")
                return
            
            # Calculate similarities using the indexed data
            similarities = []
            for doc_id, text in enumerate(text_index['texts']):
                text_words = set(re.findall(r'\w+', text.lower()))
                intersection = query_words.intersection(text_words)
                similarity = len(intersection) / len(query_words) if len(query_words) > 0 else 0
                similarities.append(similarity)
            
            # Get top results
            import numpy as np
            top_indices = np.argsort(similarities)[::-1][:3]
            
            # Display results
            found_results = False
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:
                    found_results = True
                    st.write(f"**{i+1}.** {text_index['texts'][idx][:300]}...")
                    st.write("---")
            
            if not found_results:
                st.write("No relevant content found.")
            else:
                st.info("💡 **Note:** For detailed AI answers, please wait for API quota reset or upgrade to paid plan.")
                
        else:
            # Default to simple text matching for any other embedding type
            st.info("🔍 **Searching with simple text matching...**")
            import pickle
            import re
            
            # Try to load simple index first
            if os.path.exists(os.path.join("faiss_index", "simple_index.pkl")):
                with open(os.path.join("faiss_index", "simple_index.pkl"), "rb") as f:
                    text_index = pickle.load(f)
            else:
                # Fallback: load any available text data
                st.warning("⚠️ **Using fallback text search...**")
                text_index = {'texts': ['No indexed content available']}
            
            # Simple text similarity search
            query_words = set(re.findall(r'\w+', user_question.lower()))
            if len(query_words) == 0:
                st.write("Please enter a valid question.")
                return
            
            # Calculate similarities
            similarities = []
            for doc_id, text in enumerate(text_index['texts']):
                text_words = set(re.findall(r'\w+', text.lower()))
                intersection = query_words.intersection(text_words)
                similarity = len(intersection) / len(query_words) if len(query_words) > 0 else 0
                similarities.append(similarity)
            
            # Get top results
            import numpy as np
            top_indices = np.argsort(similarities)[::-1][:3]
            
            # Display results
            found_results = False
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:
                    found_results = True
                    st.write(f"**{i+1}.** {text_index['texts'][idx][:300]}...")
                    st.write("---")
            
            if not found_results:
                st.write("No relevant content found.")
            else:
                st.info("💡 **Note:** For detailed AI answers, please wait for API quota reset or upgrade to paid plan.")
        
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")




def main():
    st.set_page_config("GenRAG-PDF")
    st.header("Chat with multiple PDF files using R A G")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("❌ Please upload at least one PDF file!")
            else:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("✅ Done! You can now ask questions about your PDFs.")
                        else:
                            st.error("❌ Failed to extract text from PDF files")
                    except Exception as e:
                        if "quota" in str(e).lower() or "429" in str(e):
                            st.error("🚫 **API Quota Exceeded!**\n\nYour Google Gemini API free tier quota has been reached. Please:\n- Wait 24 hours for quota reset, or\n- Upgrade to a paid plan, or\n- Try again later")
                        else:
                            st.error(f"❌ Processing failed: {str(e)}")



if __name__ == "__main__":
    main()