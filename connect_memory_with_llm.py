import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Database path (moved to top level for import)
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Setup Ollama LLM (Free, local model)
def load_llm():
    print("🔄 Loading Ollama model (llama2:7b)...")
    try:
        llm = Ollama(
            model="llama2:7b",  # Your downloaded model
            temperature=0.1,    # Low temperature for focused answers
            verbose=False       # Set True for debug info
        )
        # Quick test
        test_response = llm.invoke("Hello")
        print("✅ Ollama loaded successfully!")
        print(f"   Test: {test_response[:50]}...")
        return llm
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("💡 Ensure 'ollama serve' is running in another terminal.")
        raise e  # Changed from exit(1) to raise e for better error handling

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_faiss_db():
    """Load FAISS database - separated into function for better modularity"""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ FAISS database loaded successfully!")
        return db
    except Exception as e:
        print(f"❌ Error loading FAISS database: {e}")
        print("💡 Run 'python create_memory_for_llm.py' first to build the vector store from your PDFs!")
        raise e  # Changed from exit(1) to raise e

def create_qa_chain():
    """Create QA chain with all components"""
    print("🔄 Creating QA chain...")
    
    # Load components
    llm = load_llm()
    db = load_faiss_db()
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    
    print("✅ QA chain created successfully!")
    return qa_chain

def run_interactive_mode():
    """Run the interactive command-line mode"""
    # Create QA chain
    qa_chain = create_qa_chain()
    
    print("\n" + "="*60)
    print("🚀 FREE LOCAL CHATBOT READY! (Ollama + Llama2 7B)")
    print("💡 Powered by your local setup - no APIs needed!")
    print("💻 Type 'quit' to exit")
    print("="*60)

    # Interactive query loop
    while True:
        user_query = input("\n❓ Write Query Here: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_query:
            print("💭 Please enter a question.")
            continue
        
        try:
            print("🤔 Processing...")
            response = qa_chain.invoke({'query': user_query})
            
            print("\n" + "="*50)
            print("RESULT:")
            print(response["result"].strip())
            
            print("\nSOURCE DOCUMENTS:")
            for i, doc in enumerate(response["source_documents"], 1):
                print(f"\n{i}. {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"   📄 Metadata: {doc.metadata}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try rephrasing or check if Ollama is running.")

# Only run interactive mode if this script is run directly
if __name__ == "__main__":
    run_interactive_mode()