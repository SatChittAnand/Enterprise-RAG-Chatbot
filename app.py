import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import networkx as nx

from pypdf import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import spacy

# -------------------- LOAD MODELS --------------------

nlp = spacy.load("en_core_web_sm")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------- PDF HANDLING --------------------

def extract_text_from_pdfs(files):
    texts = []
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            texts.append(page.extract_text())
    return texts

def chunk_text(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.create_documents(texts)

# -------------------- VECTOR DB --------------------

def create_vector_store(docs):
    return FAISS.from_documents(docs, embedding_model)

# -------------------- CHATBOT (RAG) --------------------

def ask_question(vector_db, question):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY from the context below.
If not found, say "Answer not found in the documents."

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm(prompt)[0]["generated_text"]
    return response, docs

# -------------------- KNOWLEDGE GRAPH --------------------

def extract_triples(texts):
    triples = []
    for text in texts:
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]

        for i in range(len(entities) - 1):
            triples.append((entities[i], "related_to", entities[i + 1]))
    return triples

def build_graph(triples):
    G = nx.Graph()
    for s, p, o in triples:
        G.add_edge(s, o, label=p)
    return G

# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="AI PDF Chatbot + Knowledge Graph", layout="wide")
st.title("📄 AI PDF Chatbot with Knowledge Graph (Open Source)")

uploaded_files = st.file_uploader(
    "Upload multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success("PDFs uploaded successfully!")

    texts = extract_text_from_pdfs(uploaded_files)
    docs = chunk_text(texts)
    vector_db = create_vector_store(docs)

    triples = extract_triples(texts)
    graph = build_graph(triples)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💬 Chat with Documents")
        question = st.text_input("Ask a question")

        if question:
            answer, sources = ask_question(vector_db, question)
            st.write("### Answer")
            st.write(answer)

    with col2:
        st.subheader("🧠 Knowledge Graph")
        if st.button("Show Knowledge Graph"):
            plt.figure(figsize=(10, 8))
            nx.draw(
                graph,
                with_labels=True,
                node_size=700,
                font_size=8
            )
            st.pyplot(plt)
