import json
from sentence_transformers import SentenceTransformer
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline
import openai
import os
from dotenv import load_dotenv

# Load the JSON data
# @st.cache_data
def load_data(file_path="documents_new.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Initialize the Sentence Transformer model
@st.cache_resource
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model

# Initialize the OpenAI client
@st.cache_resource
def get_openai_client():
    load_dotenv()  # Load environment variables from .env

    openai.api_key = os.getenv("OPENAI_API_KEY") # Ensure you have this environment variable set
    return openai.OpenAI()

# Generate embeddings for the documents
# @st.cache_data
def generate_embeddings(data, model):
    embeddings = []
    contents = [item['question'] for item in data]
    embeddings = model.encode(contents)
    return embeddings

# Perform semantic search
def semantic_search(query, data, embeddings, model, top_n=3): # Reduced to top 1 for simplicity in RAG
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    results = []
    for i in sorted_indices[:top_n]:
        print(data[i])
        results.append(data[i]['text']) # Only need the content for RAG
    return results


# Generate a coherent answer using RAG with OpenAI (explicitly using gpt-3.5-turbo)
def generate_answer_openai(query, context, client):
    prompt = f"""You're a course teaching assistant. Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database. 
    Only use the facts from the CONTEXT. If the CONTEXT doesn't contain the answer, return "NONE"
    QUESTION: {query}

    CONTEXT: {context}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    """.strip()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {e}"


# Main Streamlit application
def main():
    st.title("Question Answering with Semantic Search and RAG")

    data = load_data()
    embedding_model = load_embedding_model()
    openai_client = get_openai_client()
    # generation_model = load_generation_model()
    embeddings = generate_embeddings(data, embedding_model)

    query = st.text_input("Ask a question about the document:")

    if query:
        relevant_contexts = semantic_search(query, data, embeddings, embedding_model)
       
        context = ""
        for doc in relevant_contexts:
            context += doc

        context = context.strip()

        st.subheader("Retrieved Context:")
        if context:
            st.info(context) # Display the top context
            answer = generate_answer_openai(query, context, openai_client)
            st.subheader("Generated Answer:")
            st.success(answer)
        else:
            st.warning("No relevant information found in the document.")

if __name__ == "__main__":
    main()