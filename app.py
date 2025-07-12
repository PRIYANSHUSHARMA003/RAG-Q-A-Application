import streamlit as st
from rag_chatbot import load_data, build_index, retrieve_chunks, generate_answer
from transformers import pipeline

st.set_page_config(page_title="Loan RAG Chatbot")

st.title("🤖 Loan Approval Chatbot (RAG with HuggingFace)")

@st.cache_resource
def setup():
    texts = load_data("loan_data.csv")
    model, index, embeddings = build_index(texts)
    generator = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
    return texts, model, index, generator

texts, embed_model, index, generator = setup()

query = st.text_input("Ask a question about the loan data:")

if query:
    with st.spinner("Thinking..."):
        top_chunks = retrieve_chunks(query, embed_model, index, texts)
        answer = generate_answer(query, top_chunks, generator)
    st.markdown(f"**Answer:** {answer}")
