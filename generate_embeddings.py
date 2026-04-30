import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils import CustomGeminiEmbeddings

def main():
    print("--- 🧠 Generating Embeddings ---")
    
    if os.path.exists("faiss_index"):
        print("Cleaning up old faiss_index...")
        shutil.rmtree("faiss_index")
        
    embeddings = CustomGeminiEmbeddings(model_name="gemini-embedding-2")

    print("Loading documents from ./inputs directory...")
    try:
        loader = DirectoryLoader("./inputs", glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        print(f"Loaded {len(docs)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    if not docs:
        print("No documents found in ./inputs. Exiting.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print("Creating FAISS vector database...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    print("Saving FAISS index locally to ./faiss_index")
    vectorstore.save_local("faiss_index")
    print("--- ✅ Generation Complete ---")

if __name__ == "__main__":
    main()
