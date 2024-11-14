import gradio as gr
import requests
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, BartForConditionalGeneration, BartTokenizer
import torch
from PyPDF2 import PdfReader
import io

# Load models for Q&A and Summarization
qa_model_name = "distilbert-base-cased-distilled-squad"
qa_tokenizer = DistilBertTokenizer.from_pretrained(qa_model_name)
qa_model = DistilBertForQuestionAnswering.from_pretrained(qa_model_name)

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


# Function to Extract Text from PDF
def extract_text_from_pdf(file_content):
    text = ""
    pdf_file = io.BytesIO(file_content)
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to Read Text from a File (local or URL)
def read_file(file_content, is_url=False):
    if is_url:
        # Handle URL input
        file_extension = file_content.split('.')[-1].lower()
        try:
            response = requests.get(file_content)
            response.raise_for_status()
            if file_extension == 'pdf':
                return extract_text_from_pdf(response.content)
            elif file_extension == 'txt':
                return response.text
            else:
                return "Unsupported file format from URL. Please use a .txt or .pdf file."
        except requests.exceptions.RequestException as e:
            return f"Error downloading the file: {e}"
    else:
        # Handle file upload
        file_content_bytes = file_content.read()
        if file_content.name.endswith('.pdf'):
            return extract_text_from_pdf(file_content_bytes)
        elif file_content.name.endswith('.txt'):
            return file_content_bytes.decode('utf-8')
        else:
            return "Unsupported file format. Please upload a .txt or .pdf file."


# Function to Split Text for Q&A
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


# Q&A Functionality
def answer_question_from_chunks(text, question):
    chunks = split_text_into_chunks(text)
    best_answer = ""
    best_score = 0

    for chunk in chunks:
        inputs = qa_tokenizer(question, chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)

        answer_start_index = torch.argmax(outputs.start_logits)
        answer_end_index = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.convert_tokens_to_string(
            qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start_index:answer_end_index])
        )
        score = outputs.start_logits[0][answer_start_index].item() + outputs.end_logits[0][answer_end_index - 1].item()

        if score > best_score:
            best_answer = answer
            best_score = score

    return best_answer if best_answer.strip() else "I'm not sure about the answer."


# Summarization Functionality
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


# Main function to process user input
def chatbot_function(file, url, action, question):
    if file is not None:
        training_data_text = read_file(file, is_url=False)
    elif url:
        training_data_text = read_file(url, is_url=True)
    else:
        return "Please upload a file or provide a URL."

    if "Error" in training_data_text or "Unsupported" in training_data_text:
        return training_data_text

    if action == "Summarize":
        return summarize_text(training_data_text)
    elif action == "Q&A" and question:
        return answer_question_from_chunks(training_data_text, question)
    else:
        return "Please enter a valid action and question if you selected Q&A."


# Set up Gradio interface
interface = gr.Interface(
    fn=chatbot_function,
    inputs=[
        gr.File(label="Upload File (.txt or .pdf)"),
        gr.Textbox(label="Or enter URL (http://... or https://...)", placeholder="URL to .txt or .pdf file"),
        gr.Radio(choices=["Summarize", "Q&A"], label="Choose an action"),
        gr.Textbox(label="Enter your question (only for Q&A)", placeholder="Type your question here")
    ],
    outputs="text",
    title="Chatbot with Q&A and Summarization",
    description="Upload a file or enter a URL to use the chatbot. Choose between Summarization or Q&A. If you select Q&A, enter a question.",
    allow_flagging="manual"
)

# Launch the interface
interface.launch()

