import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_neo4j import Neo4jGraph
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils import CustomGeminiEmbeddings

load_dotenv()

def format_docs(docs_list):
    return "\n\n".join(doc.page_content for doc in docs_list)

def get_graph_context(graph, query, llm):
    """
    Extracts entities from the user query and searches Neo4j for their relationships.
    """
    from langchain_core.prompts import PromptTemplate
    entity_prompt = PromptTemplate.from_template(
        "Extract the main technical entities (like service names, teams, or databases) from this query. Return them as a comma separated list.\nQuery: {query}\nEntities:"
    )
    entity_chain = entity_prompt | llm | StrOutputParser()
    entities_str = entity_chain.invoke({"query": query})
    
    entities = [e.strip() for e in entities_str.split(',')]
    
    graph_context = ""
    for entity in entities:
        if not entity: continue
        # Search Neo4j for the entity and its immediate neighbors
        # We use CONTAINS to do a fuzzy string match on node IDs
        cypher = f"""
        MATCH (n)-[r]->(m) 
        WHERE toLower(n.id) CONTAINS toLower('{entity}') OR toLower(m.id) CONTAINS toLower('{entity}') 
        RETURN n.id, type(r), m.id LIMIT 10
        """
        try:
            results = graph.query(cypher)
            for row in results:
                graph_context += f"{row['n.id']} --[{row['type(r)']}]-> {row['m.id']}\n"
        except Exception:
            pass
    return graph_context

def main():
    print("--- 🤖 Starting GraphRAG Chat Bot ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    embeddings = CustomGeminiEmbeddings(model_name="gemini-embedding-2")
    
    has_faiss = os.path.exists("faiss_index")
    
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    
    has_neo4j = all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD])

    if has_faiss and has_neo4j:
        print("[System: GraphRAG Mode Enabled - Using FAISS & Neo4j AuraDB]")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        graph = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USERNAME, 
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of vector context and graph context to answer the question. "
            "If the answer is not contained within the context, explicitly say 'I don't know based on the provided documents.' "
            "\n\n"
            "Graph Relationships Context:\n{graph_context}\n\n"
            "Unstructured Text Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        def rag_pipeline(user_input):
            # 1. Get FAISS Context
            docs = retriever.invoke(user_input)
            text_context = format_docs(docs)
            
            # 2. Get Graph Context
            graph_context = get_graph_context(graph, user_input, llm)
            
            # 3. Generate Answer
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({
                "context": text_context,
                "graph_context": graph_context,
                "input": user_input
            })
            
    else:
        print("[System: Standard LLM Mode - FAISS or Neo4j credentials missing]")
        system_prompt = "You are a helpful AI assistant. Answer to the best of your general knowledge."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        chain = (
            {"input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        def rag_pipeline(user_input):
            return chain.invoke(user_input)

    print("\nType your question below (or 'exit' to quit).\n")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
                
            response = rag_pipeline(user_input)
            print(f"Bot: {response}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
