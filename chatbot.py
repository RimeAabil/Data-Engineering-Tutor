import streamlit as st
import sys
import os
import time
from typing import Optional, Tuple

# Add the current directory to path to import your existing module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing functions
try:
    from connect_memory_with_llm import load_llm, set_custom_prompt, DB_FAISS_PATH
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all required packages are installed and your modules are in the correct directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system() -> Tuple[Optional[RetrievalQA], bool, str]:
    """Initialize the entire system with proper error handling"""
    try:
        with st.spinner("Initializing embedding model..."):
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        with st.spinner("Loading FAISS database..."):
            if not os.path.exists(DB_FAISS_PATH):
                return None, False, f"FAISS database not found at {DB_FAISS_PATH}. Please run 'python create_memory_for_llm.py' first."
            
            db = FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
        
        with st.spinner("Loading Ollama LLM..."):
            llm = load_llm()
        
        with st.spinner("Creating QA chain..."):
            custom_prompt = """
            Use the pieces of information provided in the context to answer the user's question.
            If you don't know the answer, just say that you don't know. Don't make up an answer.
            Don't provide anything outside the given context.
            
            Context: {context}
            Question: {question}
            
            Answer:"""
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt)}
            )
        
        return qa_chain, True, "System initialized successfully!"
        
    except Exception as e:
        error_msg = f"System initialization failed: {str(e)}"
        if "ollama" in str(e).lower():
            error_msg += "\n\n Make sure Ollama is running: 'ollama serve' in terminal"
        return None, False, error_msg

def display_system_status(success: bool, message: str):
    """Display system status with appropriate styling"""
    if success:
        st.markdown(f'<div class="status-box status-success">✅ {message}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-box status-error">❌ {message}</div>', 
                   unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Document Q&A Chatbot</h1>
        <p>Ask questions about your documents using local AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Information")
        
        # Initialize the system
        qa_chain, success, status_message = initialize_system()
        display_system_status(success, status_message)
        
        if success:
            st.success("🚀 Ready to answer questions!")
        else:
            st.error("⚠️ System not ready")
        
        st.markdown("---")
        
        # System settings
        st.header("⚙️ Settings")
        
        # Temperature control
        temperature = st.slider(
            "Response Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.1,
            help="Lower values make responses more focused and deterministic"
        )
        
        # Number of source documents
        num_sources = st.slider(
            "Number of Source Documents", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Number of relevant document chunks to retrieve"
        )
        
        # Show source documents toggle
        show_sources = st.checkbox(
            "Show Source Documents", 
            value=True,
            help="Display the source documents used for generating the answer"
        )
        
        st.markdown("---")
        
        # Instructions
        st.header("📝 How to Use")
        st.markdown("""
        1. **Type your question** in the chat input below
        2. **Press Enter** to get an answer
        3. **View sources** (if enabled) to see supporting documents
        4. **Adjust settings** in this sidebar for different behavior
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if not success:
        st.error("Please fix the system initialization issues before continuing.")
        st.info("Check the sidebar for more details about the error.")
        return
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display welcome message if no chat history
    if not st.session_state.messages:
        st.markdown("""
        <div class="chat-message bot-message">
            <h4>👋 Welcome!</h4>
            <p>I'm ready to answer questions about your documents. Just type your question below to get started!</p>
            <p><strong>Tips:</strong></p>
            <ul>
                <li>Ask specific questions for better results</li>
                <li>I can only answer based on the documents in your database</li>
                <li>Check the source documents to verify my answers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑‍💻 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show source documents if available and enabled
            if show_sources and "sources" in message:
                with st.expander(f"📄 View Source Documents ({len(message['sources'])} found)"):
                    for j, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {j}:**")
                        st.text(source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"])
                        if source.get("metadata"):
                            st.caption(f"Metadata: {source['metadata']}")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("💬 Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑‍💻 You:</strong><br>
            {prompt}
        </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤔 Thinking... This may take a moment..."):
            try:
                # Update retriever settings based on sidebar inputs
                qa_chain.retriever.search_kwargs = {'k': num_sources}
                
                # Get response from the QA chain
                response = qa_chain.invoke({'query': prompt})
                answer = response["result"].strip()
                
                # Prepare source documents
                sources = []
                for doc in response.get("source_documents", []):
                    sources.append({
                        "content": doc.page_content,
                        "metadata": getattr(doc, 'metadata', {})
                    })
                
                # Add assistant message to chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                }
                st.session_state.messages.append(assistant_message)
                
                # Display assistant response
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong><br>
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Show source documents if enabled
                if show_sources and sources:
                    with st.expander(f"📄 View Source Documents ({len(sources)} found)"):
                        for j, source in enumerate(sources, 1):
                            st.markdown(f"**Source {j}:**")
                            st.text(source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"])
                            if source.get("metadata"):
                                st.caption(f"Metadata: {source['metadata']}")
                            st.markdown("---")
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                
                # Add error to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
                
                # Show troubleshooting tips
                with st.expander("🔧 Troubleshooting Tips"):
                    st.markdown("""
                    - Make sure Ollama is running: `ollama serve`
                    - Try rephrasing your question
                    - Check if your documents are properly indexed
                    - Restart the application if issues persist
                    """)

if __name__ == "__main__":
    main()