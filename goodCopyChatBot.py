import os
import requests
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer
import torch
from PyPDF2 import PdfReader
import io
import pyttsx3  # Text-to-Speech for podcast generation
import graphviz  # For flowchart generation
import faiss  # For efficient similarity search in RAG

# Load DistilBERT model for Q&A
qa_model_name = "distilbert-base-cased-distilled-squad"
qa_tokenizer = DistilBertTokenizer.from_pretrained(qa_model_name)
qa_model = DistilBertForQuestionAnswering.from_pretrained(qa_model_name)

# Load BART model for Summarization
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Load Sentence Transformer for RAG
retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model for RAG
index = None  # FAISS index placeholder

# Function to Extract Text from PDF
def extract_text_from_pdf(file_content):
    text = ""
    pdf_file = io.BytesIO(file_content) if isinstance(file_content, bytes) else open(file_content, "rb")
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to Split Text into Manageable Chunks for Q&A
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Create FAISS index for RAG
def create_faiss_index(text):
    global index
    chunks = split_text_into_chunks(text)
    embeddings = retrieval_model.encode(chunks, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Create FAISS index
    index.add(embeddings)  # Add embeddings to the index
    return chunks

# Retrieve top-k relevant chunks using FAISS for RAG
def retrieve_relevant_chunks(query, chunks, top_k=3):
    global index
    query_embedding = retrieval_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Function to Answer a Question from Chunks of Text
def answer_question_from_chunks(text, question):
    chunks = create_faiss_index(text)  # Create FAISS index for retrieval
    relevant_chunks = retrieve_relevant_chunks(question, chunks)  # Retrieve relevant chunks
    combined_context = " ".join(relevant_chunks)

    inputs = qa_tokenizer(question, combined_context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = qa_model(**inputs)

    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits) + 1
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start_index:answer_end_index])
    )

    return answer if answer.strip() else "I'm not sure about the answer."

# Function to Split Text for Summarization
def split_text_for_summarization(text, chunk_size=1000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to Summarize Text
def summarize_chunk(chunk):
    inputs_summary = bart_tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(
        inputs_summary["input_ids"],
        max_length=50,
        min_length=20,
        num_beams=4,
        early_stopping=True
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_text(text):
    chunks = split_text_for_summarization(text, chunk_size=1000)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    return " ".join(summaries)

# Function to Convert Text to Audio (Podcast Generation)
def text_to_audio(text, filename="podcast.mp3"):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()

# Function to Create a Flowchart from Text
def text_to_flowchart(text, max_nodes=8):
    # Summarize the text to get the key points
    summarized_text = summarize_text(text)
    lines = summarized_text.split(". ")  # Split by sentence or period
    lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines

    # Limit to max_nodes for a concise flowchart
    selected_lines = lines[:max_nodes]

    graph = graphviz.Digraph(format='png', engine='dot')

    for i, line in enumerate(selected_lines):
        graph.node(f'{i}', line)  # Create a node for each line
        if i > 0:
            graph.edge(f'{i-1}', f'{i}')  # Link the nodes in sequence

    return graph

# Streamlit Interface
st.title("AI Chatbot with Q&A, Summarization, Podcast & Flowchart")

file_input_method = st.radio("Choose an input method:", ["Upload a file", "Enter a URL"])
text_content = None

if file_input_method == "Upload a file":
    file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    if file is not None:
        if file.type == "application/pdf":
            text_content = extract_text_from_pdf(file.read())
        elif file.type == "text/plain":
            text_content = file.read().decode("utf-8")
else:
    file_url = st.text_input("Enter the URL of a .txt or .pdf file:")
    if file_url:
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            if file_url.endswith(".pdf"):
                text_content = extract_text_from_pdf(response.content)
            elif file_url.endswith(".txt"):
                text_content = response.text
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading the file: {e}")

if text_content:
    action = st.radio("What would you like to do?", ["Summarize", "Ask a Question", "Generate Podcast", "Generate Flowchart"])

    if action == "Summarize":
        summary = summarize_text(text_content)
        st.subheader("Summary")
        st.write(summary)

    elif action == "Ask a Question":
        question = st.text_input("Enter your question:")
        if question:
            answer = answer_question_from_chunks(text_content, question)
            st.subheader("Answer")
            st.write(answer)

    elif action == "Generate Podcast":
        # Convert the text to an audio podcast
        text_to_audio(text_content)
        st.audio("podcast.mp3", format="audio/mp3")
        st.success("Podcast generated successfully!")

    elif action == "Generate Flowchart":
        # Create a flowchart diagram from the text
        flowchart = text_to_flowchart(text_content)
        flowchart.render('flowchart', view=True)
        st.image('flowchart.png')
        st.success("Flowchart generated successfully!")
