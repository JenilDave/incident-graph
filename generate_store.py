import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from utils import CustomGeminiEmbeddings

load_dotenv()

def main():
    print("--- 🧠 Generating Embeddings & Graph Store ---")
    
    if os.path.exists("faiss_index"):
        print("Cleaning up old faiss_index...")
        shutil.rmtree("faiss_index")
        
    embeddings = CustomGeminiEmbeddings(model_name="gemini-embedding-2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # Neo4j setup
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        print("Neo4j credentials not found in .env. Please configure them.")
        return

    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )

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

    print("Extracting Graph Nodes and Relationships (with rate limit protection)...")
    import time
    from tenacity import retry, wait_exponential, stop_after_attempt

    @retry(wait=wait_exponential(multiplier=2, min=10, max=60), stop=stop_after_attempt(10))
    def process_chunk_with_retry(llm_transformer, split):
        return llm_transformer.convert_to_graph_documents([split])

    # LLMGraphTransformer extracts structured relationships out of unstructured text
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    graph_documents = []
    for i, split in enumerate(splits):
        print(f"   Processing chunk {i+1} of {len(splits)}...")
        try:
            doc = process_chunk_with_retry(llm_transformer, split)
            graph_documents.extend(doc)
            # Sleep aggressively to respect Gemini API Free Tier limits (15 RPM)
            if i < len(splits) - 1:
                print("   Sleeping for 15 seconds to avoid rate limits...")
                time.sleep(15)
        except Exception as e:
            print(f"   Failed to process chunk {i+1} after all retries: {e}")
    
    print("Saving relationships to Neo4j AuraDB...")
    graph.add_graph_documents(
        graph_documents, 
        baseEntityLabel=True, 
        include_source=True
    )

    print("--- ✅ Generation Complete ---")

if __name__ == "__main__":
    main()
