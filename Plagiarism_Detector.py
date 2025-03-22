#!/usr/bin/env python3
"""
plagiarism_checker.py

A terminal-based AI plagiarism detection system that processes answer sheets (PDFs/images)
to detect:
  - High similarity plagiarism using TF-IDF + cosine similarity.
  - Partial plagiarism using sentence-level matching.
  - Paraphrasing using the Google Gemini API.
  
Usage:
    python plagiarism_checker.py <folder_path> [--csv output.csv]

Ensure you have installed required packages:
    pip install pytesseract opencv-python PyPDF2 nltk scikit-learn google-generativeai numpy
"""

import os
import sys
import cv2
import csv
import pytesseract
import PyPDF2
import traceback
import numpy as np
import nltk
import concurrent.futures

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# For paraphrasing detection using Gemini API
import google.generativeai as genai

# Configure Google Gemini API (ensure you have your API key correctly set up)
genai.configure(api_key="AIzaSyAQvW-7i3jnNu5qwolDOPV9q2HhdkKtrAU")
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Download necessary NLTK data if not already present
nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def extract_text_from_pdf(file_path):
    """Extract text from a PDF using PyPDF2."""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_image(file_path):
    """Extract text from an image using Tesseract and OpenCV."""
    text = ""
    try:
        image = cv2.imread(file_path)
        if image is not None:
            text = pytesseract.image_to_string(image)
        else:
            print(f"Could not read image file {file_path}")
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
    return text

def preprocess_text(text):
    """Preprocess text: lowercase, remove stopwords and punctuation, and tokenize."""
    try:
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum() and t not in STOPWORDS]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error in preprocessing text: {e}")
        return text

def process_file(file_path):
    """Extract and preprocess text from a file (PDF or image)."""
    ext = os.path.splitext(file_path)[1].lower()
    raw_text = ""
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        raw_text = extract_text_from_image(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
    processed = preprocess_text(raw_text)
    return processed, raw_text  # returning both processed and raw for sentence splitting later

def compute_document_similarity(doc_texts, file_names):
    """Compute cosine similarity between documents using TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    similar_pairs = []
    num_docs = len(file_names)
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            similarity = sim_matrix[i, j]
            if similarity > 0.75:
                similar_pairs.append((file_names[i], file_names[j], similarity))
    return similar_pairs

def sentence_level_plagiarism(raw_text1, raw_text2, threshold=0.8, min_matches=3):
    """Detect partial plagiarism by comparing sentences between two documents."""
    sentences1 = sent_tokenize(raw_text1)
    sentences2 = sent_tokenize(raw_text2)
    vectorizer = TfidfVectorizer().fit(sentences1 + sentences2)
    tfidf1 = vectorizer.transform(sentences1)
    tfidf2 = vectorizer.transform(sentences2)
    
    count = 0
    # Compute pairwise cosine similarity for sentences
    sim = cosine_similarity(tfidf1, tfidf2)
    for i in range(sim.shape[0]):
        if np.any(sim[i] >= threshold):
            count += 1
    return count >= min_matches

def check_paraphrasing(text1, text2):
    """
    Use Google Gemini API to detect paraphrasing between two texts.
    Returns True if paraphrasing is detected, False otherwise.
    Fallback: if API call fails, returns None.
    """
    try:
        prompt = (f"Determine if the following texts contain paraphrased content. "
                  f"Text1: '''{text1[:500]}''' Text2: '''{text2[:500]}''' "
                  f"Respond with 'Yes' or 'No'.")
        response = model.generate(prompt)
        answer = response.result.strip().lower()
        if 'yes' in answer:
            return True
        else:
            return False
    except Exception as e:
        print(f"Gemini API error: {e}. Skipping paraphrasing check for these documents.")
        return None

def main(folder_path, csv_output=None):
    # Gather all eligible file paths
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                  if os.path.splitext(f)[1].lower() in [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]
    if not file_paths:
        print("No supported files found in the folder.")
        return

    print(f"Processing {len(file_paths)} files...")

    # Process files concurrently
    processed_texts = {}
    raw_texts = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, fp): fp for fp in file_paths}
        for future in concurrent.futures.as_completed(future_to_file):
            fp = future_to_file[future]
            try:
                processed, raw = future.result()
                processed_texts[fp] = processed
                raw_texts[fp] = raw
            except Exception as exc:
                print(f"{fp} generated an exception: {exc}")
    
    # High similarity plagiarism detection
    doc_names = list(processed_texts.keys())
    docs = [processed_texts[name] for name in doc_names]
    high_similar_pairs = compute_document_similarity(docs, doc_names)
    
    print("\n--- High Similarity Plagiarism (Cosine Similarity > 75%) ---")
    for f1, f2, sim in high_similar_pairs:
        print(f"{f1} <--> {f2} : Similarity = {sim*100:.2f}%")
    
    # Partial plagiarism detection (sentence-level)
    print("\n--- Partial Plagiarism Detection (Sentence-level) ---")
    partial_plagiarism_results = []
    for i in range(len(doc_names)):
        for j in range(i + 1, len(doc_names)):
            f1, f2 = doc_names[i], doc_names[j]
            if sentence_level_plagiarism(raw_texts[f1], raw_texts[f2]):
                print(f"Partial plagiarism detected between {f1} and {f2}")
                partial_plagiarism_results.append((f1, f2))
    
    # Paraphrasing detection using Gemini API (only for high similarity pairs)
    print("\n--- Paraphrasing Detection (Using Gemini API) ---")
    paraphrasing_results = []
    for f1, f2, sim in high_similar_pairs:
        # Only check if documents are suspicious
        result = check_paraphrasing(raw_texts[f1], raw_texts[f2])
        if result is True:
            print(f"Paraphrasing detected between {f1} and {f2}")
            paraphrasing_results.append((f1, f2, True))
        elif result is False:
            print(f"No paraphrasing detected between {f1} and {f2}")
            paraphrasing_results.append((f1, f2, False))
        else:
            print(f"Paraphrasing check skipped for {f1} and {f2}")

    # Optional CSV output
    if csv_output:
        try:
            with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Document1', 'Document2', 'Cosine Similarity', 'Partial Plagiarism', 'Paraphrasing']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # Combine high similarity pairs into one record (and check partial & paraphrasing results)
                for f1, f2, sim in high_similar_pairs:
                    partial = "Yes" if (f1, f2) in partial_plagiarism_results or (f2, f1) in partial_plagiarism_results else "No"
                    paraphrase_flag = next((str(flag) for (a, b, flag) in paraphrasing_results if (a == f1 and b == f2) or (a == f2 and b == f1)), "N/A")
                    writer.writerow({
                        'Document1': f1,
                        'Document2': f2,
                        'Cosine Similarity': f"{sim*100:.2f}%",
                        'Partial Plagiarism': partial,
                        'Paraphrasing': paraphrase_flag
                    })
            print(f"\nCSV output saved to {csv_output}")
        except Exception as e:
            print(f"Error writing CSV output: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plagiarism_checker.py <folder_path> [--csv output.csv]")
        sys.exit(1)
    folder = sys.argv[1]
    csv_file = None
    if "--csv" in sys.argv:
        try:
            csv_index = sys.argv.index("--csv")
            csv_file = sys.argv[csv_index + 1]
        except IndexError:
            print("Please provide a file name for CSV output after '--csv'")
            sys.exit(1)
    try:
        main(folder, csv_file)
    except Exception:
        traceback.print_exc()
