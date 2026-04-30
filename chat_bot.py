import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils import CustomGeminiEmbeddings

def format_docs(docs_list):
    return "\n\n".join(doc.page_content for doc in docs_list)

def main():
    print("--- 🤖 Starting Chat Bot ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    embeddings = CustomGeminiEmbeddings(model_name="gemini-embedding-2")
    
    has_faiss = os.path.exists("faiss_index")
    
    if has_faiss:
        print("[System: RAG Mode Enabled - Using FAISS Index]")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If the answer is not contained within the context, explicitly say 'I don't know based on the provided documents.' "
            "Do not hallucinate or use outside knowledge to answer context-specific questions."
            "\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        print("[System: Standard LLM Mode - FAISS Index NOT Found]")
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

    print("\nType your question below (or 'exit' to quit).\n")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
                
            response = chain.invoke(user_input)
            print(f"Bot: {response}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
