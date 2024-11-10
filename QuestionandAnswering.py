import os
import requests
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from PyPDF2 import PdfReader
import io

# Load DistilBERT model for Q&A
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Function to Extract Text from PDF
def extract_text_from_pdf(file_content):
    """Extracts text from a PDF file or byte content."""
    text = ""
    pdf_file = io.BytesIO(file_content) if isinstance(file_content, bytes) else open(file_content, "rb")
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to Read Text from a File (local or URL)
def read_file(file_path):
    """Reads text from a local file or URL."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            response = requests.get(file_path)
            response.raise_for_status()
            if file_extension == '.pdf':
                return extract_text_from_pdf(response.content)
            elif file_extension == '.txt':
                return response.text
            else:
                return "Unsupported file format from URL. Please use a .txt or .pdf file."
        except requests.exceptions.RequestException as e:
            return f"Error downloading the file: {e}"
    elif os.path.exists(file_path):
        try:
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            elif file_extension == '.pdf':
                return extract_text_from_pdf(file_path)
            else:
                return "Unsupported file format. Please use a .txt or .pdf file."
        except Exception as e:
            return f"Error reading the file: {e}"
    else:
        return "Error: File not found."

# Function to split text into manageable chunks (to fit model input size)
def split_text_into_chunks(text, chunk_size=500):
    """Splits the text into chunks that are no longer than the specified chunk_size."""
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

# Function to answer a question using the chunks of text
def answer_question_from_chunks(text, question):
    """Generates an answer to the question based on chunks of the training data."""
    chunks = split_text_into_chunks(text)
    best_answer = ""
    best_score = 0

    for chunk in chunks:
        inputs = tokenizer(question, chunk, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract the answer span
        answer_start_index = torch.argmax(outputs.start_logits)
        answer_end_index = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start_index:answer_end_index])
        )
        
        # Score based on the logits for start and end positions
        score = outputs.start_logits[0][answer_start_index].item() + outputs.end_logits[0][answer_end_index - 1].item()

        if score > best_score:
            best_answer = answer
            best_score = score

    return best_answer if best_answer.strip() else "I'm not sure about the answer."

# Main Function to Handle User Input
def main():
    while True:
        print("Welcome to the Q&A chatbot!")

        input_method = input("Do you want to (1) Paste your training data or (2) Provide a link to it? Please enter 1 or 2: ")

        if input_method == '1':
            training_data_text = input("Please paste your training data here: ")
        elif input_method == '2':
            file_url = input("Please provide the link to your training data (e.g., a URL to a .txt or .pdf file): ")
            training_data_text = read_file(file_url)
        else:
            print("Invalid choice. Please restart the program and choose 1 or 2.")
            return

        if "Error" in training_data_text or "Unsupported" in training_data_text:
            print(training_data_text)
            return

        action = input("Do you want to ask a question about the training data? Please enter 'yes' or 'no': ")

        if action.lower() == 'yes':
            question = input("Please enter your question about the training data: ")
            answer = answer_question_from_chunks(training_data_text, question)
            print("\nAnswer:")
            print(answer)
        else:
            print("Invalid choice. Please restart the program and choose 'yes' or 'no'.")

        continue_choice = input("\nDo you want to continue using the program? (yes/no): ").lower()
        if continue_choice != 'yes':
            print("Exiting the program. Goodbye!")
            break

# Run the main function
if __name__ == "__main__":
    main()
