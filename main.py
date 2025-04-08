import streamlit as st
import wikipedia
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Set Streamlit page config
st.set_page_config(page_title="WIKI RAG BOT", page_icon="📚")

# Title
st.markdown("<h1 style='text-align: center;'>📚 Mini Wikipedia RAG Q&A Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Built using Wikipedia + Hugging Face transformers</p>", unsafe_allow_html=True)


# Load embedding model and QA pipeline
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return embedder, qa_pipeline

embedder, qa = load_models()

# Input box
user_question = st.text_input("🔍 Ask me a question:")

if user_question:
    try:
        # Search Wikipedia
        st.write("📥 Searching Wikipedia...")
        search_results = wikipedia.search(user_question)
        if not search_results:
            st.error("No relevant Wikipedia pages found.")
        else:
            top_title = search_results[0]
            summary = wikipedia.summary(top_title, sentences=10)
            st.write(f"📄 Retrieved summary from: **{top_title}**")
            st.info(summary)

            # Chunk the summary
            sentences = summary.split('. ')
            chunks = [". ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]

            # Embed chunks and user question
            chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
            question_embedding = embedder.encode(user_question, convert_to_tensor=True)

            # Semantic similarity to find best chunk
            scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
            top_chunk = chunks[scores.argmax().item()]

            st.write("📌 Most relevant chunk selected for answering:")
            st.success(top_chunk)

            # Feed to QA model
            st.write("💬 Generating answer...")
            answer = qa(question=user_question, context=top_chunk)

            st.subheader("🧠 Answer:")
            st.success(answer['answer'])

    except Exception as e:
        st.error(f"An error occurred: {e}")
