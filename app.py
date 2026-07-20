
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import google.generativeai as genai
import faiss
import numpy as np
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-2.5-flash"

st.set_page_config(
    page_title="PDF RAG Chatbot with Translator",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot + 🌐 Translator")

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

with st.sidebar:
    st.header("Controls")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.button("Reset"):
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.pdf_loaded = False
        st.session_state.chat_history = []
        st.session_state.last_answer = ""

        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

        st.rerun()

if uploaded_pdf is not None and not st.session_state.pdf_loaded:
    with st.spinner("Processing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=51
        )

        chunks = splitter.split_documents(documents)

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = embedding_model.encode(texts)

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(
            np.array(embeddings).astype("float32")
        )

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.pdf_loaded = True

    st.success(
        f"PDF processed successfully!\n\n"
        f"Pages: {len(documents)}\n"
        f"Chunks: {len(chunks)}"
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pdf_loaded:
    user_query = st.chat_input(
        "Ask a question about the PDF..."
    )

    if user_query:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_query
            }
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    query_embedding = embedding_model.encode(
                        [user_query]
                    )

                    D, I = st.session_state.index.search(
                        np.array(query_embedding).astype("float32"),
                        k=5
                    )

                    context = ""

                    for idx in I[0]:
                        context += (
                            st.session_state.chunks[idx]
                            .page_content
                            + "\n\n"
                        )

                    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context below.

If the answer is not present in the context, say:
"I could not find the answer in the uploaded PDF."

Context:
{context}

Question:
{user_query}

Provide a clear answer in bullet points.
"""

                    model = genai.GenerativeModel(
                        MODEL_NAME
                    )

                    response = model.generate_content(
                        prompt
                    )

                    answer = response.text

                except Exception as e:
                    answer = f"Error: {str(e)}"

                st.markdown(answer)

                st.session_state.last_answer = answer

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

else:
    st.info(
        "👈 Upload a PDF from the sidebar to start."
    )

st.divider()

st.subheader("🌐 Translate Any Text")

languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml"
}

translate_text = st.text_area(
    "Enter text to translate"
)

selected_language = st.selectbox(
    "Translate to",
    list(languages.keys()),
    key="manual"
)

if st.button("Translate Text"):
    if translate_text.strip():
        try:
            translated = GoogleTranslator(
                source="auto",
                target=languages[selected_language]
            ).translate(translate_text)

            st.success("Translated Text")
            st.write(translated)

        except Exception as e:
            st.error(
                f"Translation Error: {str(e)}"
            )
    else:
        st.warning(
            "Please enter some text."
        )

st.divider()

st.subheader("🤖 Translate Latest Chatbot Answer")

if st.session_state.last_answer:
    chatbot_language = st.selectbox(
        "Select language",
        list(languages.keys()),
        key="chatbot"
    )

    if st.button("Translate Chatbot Answer"):
        try:
            translated_answer = GoogleTranslator(
                source="auto",
                target=languages[chatbot_language]
            ).translate(
                st.session_state.last_answer
            )

            st.success(
                f"Translated Answer ({chatbot_language})"
            )

            st.write(translated_answer)

        except Exception as e:
            st.error(
                f"Translation Error: {str(e)}"
            )

else:
    st.info(
        "Ask the chatbot a question first."
    )

