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

# Set the page title
st.set_page_config(page_title="Verba", page_icon=":speech_balloon:")

# Styling the header, subtitle, search bar, and progress bar
st.markdown(
    """
    <style>
    .header {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        padding: 10px 0;
        border-bottom: 2px solid #ddd;
    }
    .logo {
        height: 50px;
        margin-right: 165px;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        margin: 0;
        color: blue;
    }
    .subtitle {
        font-size: 20px;
        color: blue;
        text-align: left;
        margin-left: 10px;
        margin-top: 15px;
    }
    .learning-info {
        font-size: 16px;
        color: #666;
        text-align: left;
        margin-left: 10px;
        margin-top: 5px;
    }
    .search-bar {
        width: 100%;
        max-width: 500px;
        margin: 10px auto 20px auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
        color: #666;
        background-color: #f9f9f9;
    }
    .progress-bar-container {
        width: 100%;
        margin: 20px 0;
        text-align: center;
    }
    .progress-bar {
        width: 65%;
        background-color: blue;
        color: white;
        padding: 5px;
        border-radius: 5px;
    }
    .user-profile {
        text-align: right;
        position: absolute;
        top: 10px;
        right: 10px;
    }
    .user-profile img {
        height: 50px;
        border-radius: 50%;
        margin-bottom: 5px;
    }
    .learning-points {
        font-size: 14px;
        color: #555;
    }
    .video-box-container {
        display: flex;
        justify-content: space-evenly;  /* Equally space out the boxes */
        margin-top: 20px;
    }
    .video-box {
        background-color: #f0f0f0;  /* Light grey color */
        width: 30%;  /* Each box takes up 30% of the container's width */
        padding: 20px;
        text-align: center;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-size: 18px;
        color: #333;
    }

        .last-login {
        font-size: 14px;
        color: #555;  /* Same grey as learning points */
        text-align: left;
        margin-top: 10px;
    }
    
    .continue-learning, .learning-info {
        font-size: 16px;
        color: blue;
        margin-top: 20px;
        text-align: center;
    }

        .blue-subtitle {
        font-size: 18px;
        color: blue;
        text-align: center;
        margin-top: 20px;
    }

    .footer {
        font-size: 18px;
        color: blue;
        text-align: center;
        margin-top: 30px;
    }

    </style>

    </style>
    <div class="header">
        <img src="https://lsminsurance.ca/images/RBC-insurance-logo-1.jpg" alt="RBC Insurance Logo" class="logo">
        <h1 class="title">Verba</h1>
    </div>
    <p class="subtitle">Verba Learning Centre</p>
    <p class="learning-info">Last logged in November 27, 2024</p>
    <div class="search-bar">Search learning library</div>
    <div class="progress-bar-container">
        <div class="progress-bar">65% complete</div>
    </div>
    <div class="user-profile">
        <img src="https://storage.needpix.com/rsynced_images/blank-profile-picture-973460_1280.png" alt="User Profile">
        <p class="learning-points">91 learning points</p>
    </div>
    <p class="continue-learning">Continue with your learning videos...</p>
    <div class="video-box-container">
        <div class="video-box">Video One</div>
        <div class="video-box">Video Two</div>
        <div class="video-box">Video Three</div>
    </div>
    <p class="blue-subtitle">Verba Chatbot</p>

    """,
    unsafe_allow_html=True,
)


# Display the main interface options below the header
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
            if response.headers["Content-Type"] == "application/pdf":
                text_content = extract_text_from_pdf(response.content)
            elif "text" in response.headers["Content-Type"]:
                text_content = response.text
        except Exception as e:
            st.error(f"Failed to fetch the file. Error: {e}")

if text_content:
   ## st.text_area("Extracted Content", text_content[:2000], height=200)
    task = st.radio("Select a task:", ["Q&A", "Summarization", "Generate Podcast", "Create Flowchart"])

    if task == "Q&A":
        question = st.text_input("Enter your question:")
        if question:
            answer = answer_question_from_chunks(text_content, question)
            st.write("Answer:", answer)
    elif task == "Summarization":
        summary = summarize_text(text_content)
        st.write("Summary:", summary)
    elif task == "Generate Podcast":
        podcast_filename = "podcast.mp3"
        summarized_text = summarize_text(text_content)  # Use summarized text
        text_to_audio(text_content, podcast_filename)
        with open(podcast_filename, "rb") as file:
            st.audio(file.read(), format="audio/mp3")
    elif task == "Create Flowchart":
        summarized_text = summarize_text(text_content)  # Use summarized text
        flowchart = text_to_flowchart(summarized_text)
        st.graphviz_chart(flowchart.source)

# Add a profile picture underneath the "View your community members" subtitle
st.markdown(
    """
    <div class="footer">View your community members</div>
    <div class="profile-pictures-container">
        <img src="https://storage.needpix.com/rsynced_images/blank-profile-picture-973460_1280.png" alt="Community Member 1" class="profile-picture">
        <img src="https://storage.needpix.com/rsynced_images/blank-profile-picture-973460_1280.png" alt="Community Member 2" class="profile-picture">
        <img src="https://storage.needpix.com/rsynced_images/blank-profile-picture-973460_1280.png" alt="Community Member 3" class="profile-picture">
        <img src="https://storage.needpix.com/rsynced_images/blank-profile-picture-973460_1280.png" alt="Community Member 4" class="profile-picture">
        <img src="https://storage.needpix.com/rsynced_images/blank-profile-picture-973460_1280.png" alt="Community Member 5" class="profile-picture">
        <img src="https://storage.needpix.com/rsynced_images/blank-profile-picture-973460_1280.png" alt="Community Member 6" class="profile-picture">
    </div>
    <style>
        .profile-pictures-container {
            display: flex;
            justify-content: space-evenly;  /* Equally space out the profile pictures */
            margin-top: 10px;
        }
        .profile-picture {
            height: 50px;
            width: 50px;
            border-radius: 50%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

