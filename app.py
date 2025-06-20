import os
import streamlit as st
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

def get_gemini_api_key():
    """
    Gets the Gemini API key from environment variables.
    Raises an error if the key is not found.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔴 **Error:** Your GOOGLE_API_KEY is not set!")
        st.info("Please create a `.env` file and set the key, or set it as an environment variable to continue.")
        st.stop()
    return api_key

@st.cache_resource
def get_embeddings():
    """
    Loads and caches the HuggingFace embeddings model to avoid reloading.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(_api_key):
    """
    Loads and caches the Language Model.
    Removed the deprecated 'convert_system_message_to_human' parameter.
    """
    return ChatGoogleGenerativeAI(google_api_key=_api_key, model="gemini-1.5-flash-latest",
                                         temperature=0.4)

def create_conversational_chain(vector_store, llm):
    """
    Creates the conversational chain for Q&A.
    """
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="PageSense", page_icon="🤖", layout="wide")

    st.markdown("""
        <style>
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            .st-emotion-cache-16txtl3 {
                padding: 2rem 1rem;
                background-color: #0E1117;
            }
            .st-emotion-cache-1y4p8pa {
                width: 100%;
                padding: 2rem 1rem 10rem;
                max-width: 80rem;
            }
            .st-chat-message {
                background-color: #262730;
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.5);
            }
            h1, h2, h3, h4, h5, h6 {
                color: #FAFAFA;
            }
        </style>
    """, unsafe_allow_html=True)


    api_key = get_gemini_api_key()
    llm = get_llm(api_key)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None

    with st.sidebar:
        st.title("📄 Document Hub")
        st.markdown("Upload a document and the AI will answer questions based on its content.")

        uploaded_file = st.file_uploader(
            "Upload your PDF, DOCX, or TXT file",
            type=["pdf", "docx", "txt"],
            help="The chatbot will use this document as its knowledge base."
        )

        if uploaded_file:
            if uploaded_file.name != st.session_state.uploaded_file_name:
                with st.spinner("Analyzing document... This may take a moment."):
                    try:
                        temp_dir = "temp_docs"
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        ext = os.path.splitext(uploaded_file.name)[-1].lower()
                        if ext == '.pdf': loader = PyPDFLoader(temp_path)
                        elif ext == '.docx': loader = Docx2txtLoader(temp_path)
                        elif ext == '.txt': loader = TextLoader(temp_path, encoding='utf-8')
                        else: raise ValueError("Unsupported file type")
                        
                        documents = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        docs = text_splitter.split_documents(documents)
                        embeddings = get_embeddings()
                        vector_store = FAISS.from_documents(docs, embeddings)

                        st.session_state.conversation = create_conversational_chain(vector_store, llm)
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.session_state.chat_history = []
                        st.success("Document analyzed successfully! You can start chatting.")
                        os.remove(temp_path)

                    except Exception as e:
                        st.error(f"Failed to process document: {e}")
                        st.session_state.conversation = None
                        st.session_state.uploaded_file_name = None


    st.title("🤖 Gemini AI Chatbot")
    st.markdown("Ask me anything! If you've uploaded a document, I'll use it for context.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                if st.session_state.conversation:
                    history_for_chain = []
                    chat_history_for_chain = st.session_state.chat_history[:-1] 
                    for i in range(0, len(chat_history_for_chain), 2):
                        user_msg = chat_history_for_chain[i]
                        ai_msg = chat_history_for_chain[i+1]
                        if user_msg["role"] == "user" and ai_msg["role"] == "assistant":
                            history_for_chain.append((user_msg["content"], ai_msg["content"]))
                    
                    result = st.session_state.conversation.invoke(
                        {"question": prompt, "chat_history": history_for_chain}
                    )
                    response = result["answer"]
                else:
                    response = llm.invoke(prompt).content

                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
