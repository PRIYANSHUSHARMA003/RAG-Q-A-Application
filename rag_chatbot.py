import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def load_data(file_path="loan_data.csv"):
    df = pd.read_csv(file_path)
    df.fillna("Unknown", inplace=True)
    texts = [" | ".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()]
    return texts

def build_index(text_chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index, embeddings

def retrieve_chunks(query, embed_model, index, texts, k=5):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [texts[i] for i in indices[0]]

def generate_answer(query, retrieved_chunks, generator):
    context = "\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=200, do_sample=False)
    return response[0]["generated_text"].split("Answer:")[-1].strip()
