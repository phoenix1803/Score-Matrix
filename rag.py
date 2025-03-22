import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks
from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings
import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index
def retrieve_relevant_chunks(question, index, chunks, embeddings_model, top_k=3):
    question_embedding = embeddings_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks
import google.generativeai as genai

genai.configure(api_key="")
def generate_answer(question, relevant_chunks, model = genai.GenerativeModel("gemini-1.5-flash-latest")):
    context = " ".join(relevant_chunks)
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    
    response = model.generate_content(prompt)
    return response.text
def rag_system(pdf_path, question):
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Split text into chunks
    chunks = split_text_into_chunks(text)
    
    # Step 3: Generate embeddings
    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = generate_embeddings(chunks)
    
    # Step 4: Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Step 5: Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(question, index, chunks, embeddings_model)
    print("hehe")
    
    # Step 6: Generate answer using a paid model
    answer = generate_answer(question, relevant_chunks)
    
    return answer
pdf_path = "human-values.pdf"
question = "What is the main topic of this book?"
answer = rag_system(pdf_path, question)
print("Answer:", answer)
