import streamlit as st
import spacy
import os
from docx import Document
from pypdf import PdfReader # Updated from PyPDF2 to pypdf for modern usage
import ollama
import re # For regular expressions (basic text cleaning)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np # Used by scikit-learn

# --- Configuration ---
OLLAMA_MODEL = "gemma:2b" # Ensure this model is downloaded via `ollama run gemma:2b`
TEMP_UPLOAD_DIR = "temp_uploads"

# Create temp directory if it doesn't exist to store uploaded files temporarily
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- Load NLP Model (spaCy) ---
# Load a small English model for basic NLP tasks
# Run 'python -m spacy download en_core_web_sm' in your terminal if you haven't
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal (with your virtual environment active).")
    st.stop() # Stop the app if model is not found, as core functionality won't work

# --- Helper Functions ---

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or "" # Use .extract_text() and handle None for empty pages
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}. Please ensure it's a valid PDF.")
        return ""

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}. Please ensure it's a valid DOCX.")
        return ""

def preprocess_text_spacy(text):
    """
    Processes text using spaCy for basic cleaning and tokenization.
    Removes stopwords and non-alphabetic characters.
    """
    if not text:
        return ""
    doc = nlp(text.lower()) # Process text in lowercase
    # Filter out stopwords (common words like 'the', 'is'), punctuation, and non-alphabetic tokens.
    # Use token.lemma_ to get the base form of the word (e.g., 'running' -> 'run').
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens) # Join processed tokens back into a single string

def calculate_cosine_similarity(text1, text2):
    """Calculates cosine similarity between two preprocessed text documents."""
    if not text1 or not text2:
        return 0.0 # Return 0 similarity if one or both texts are empty

    # TF-IDF Vectorizer: Converts text into numerical vectors where words are weighted
    # based on their importance (how frequently they appear in a document but rarely in others).
    vectorizer = TfidfVectorizer() # Stopwords were already removed in preprocess_text_spacy

    # Fit the vectorizer to both texts and transform them into TF-IDF vectors.
    # The fit_transform method expects an iterable (like a list) of documents.
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate cosine similarity between the first (resume) and second (job description) vector.
    # tfidf_matrix[0:1] means "take the first row as a 2D array"
    # tfidf_matrix[1:2] means "take the second row as a 2D array"
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return cosine_sim

# Use Streamlit's caching mechanism for this function.
# This means if the 'prompt' is the same, Streamlit won't re-run the Ollama call,
# making the app faster on subsequent identical requests.
@st.cache_data
def analyze_with_ollama(prompt):
    """Sends a prompt to the local Ollama model and returns the response."""
    st.info(f"Asking {OLLAMA_MODEL} for analysis... This might take a moment (first run might be slower).")
    try:
        # Communicate with the Ollama server running locally
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': prompt}, # The prompt is sent as a user message
        ], stream=False) # stream=False means wait for the full response

        return response['message']['content'] # Extract the AI's response text
    except Exception as e:
        # Catch potential errors like Ollama server not running or model not found
        st.error(f"Error communicating with Ollama: {e}. Make sure Ollama is running and '{OLLAMA_MODEL}' is downloaded.")
        st.warning(f"You can download the model by running `ollama run {OLLAMA_MODEL}` in your terminal.")
        return None

# --- Streamlit UI Layout ---

# Set basic page configuration
# Removed 'icon' argument due to potential compatibility issues with older Streamlit versions.
# If you verified your Streamlit is 1.10.0+ and want an icon, you can re-add: icon="🤖"
st.set_page_config(page_title="AI Resume Analyzer & Job Match", layout="wide")

st.title("🤖 AI Resume Analyzer & Job Match Platform")
st.markdown("""
Welcome! Upload your resume and paste a job description to get instant feedback and see how well you match.
All powerful AI analysis runs locally on your computer using Ollama for privacy and speed!
""")

# --- Input Section ---
st.header("1. Upload Your Resume")
# File uploader widget: accepts PDF and DOCX
uploaded_resume_file = st.file_uploader("Choose a resume file (PDF or DOCX)", type=["pdf", "docx"], key="resume_uploader")

st.header("2. Paste the Job Description")
# Text area widget for job description input
job_description = st.text_area("Paste the full job description here:", height=300, key="job_desc_input",
                               placeholder="e.g., 'We are looking for a highly motivated Software Engineer with expertise in Python, machine learning, and cloud platforms...'")

# Button to trigger analysis
analyze_button = st.button("✨ Analyze & Match My Resume!")

# --- Processing and Output ---
# This block runs only when the button is clicked AND there's at least one input
if analyze_button:
    # Check if both inputs are provided
    if not uploaded_resume_file:
        st.warning("☝️ Please upload your resume to get started.")
        st.stop() # Stop further execution if no resume
    if not job_description:
        st.warning("✍️ Please paste the job description you want to match against.")
        st.stop() # Stop further execution if no job description

    # Use a spinner to show that processing is happening
    with st.spinner("🚀 Analyzing your resume and job description... This might take a moment."):
        resume_text = ""
        # Save the uploaded file temporarily to disk to allow pypdf/python-docx to read it
        temp_resume_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_resume_file.name)
        with open(temp_resume_path, "wb") as f:
            f.write(uploaded_resume_file.getbuffer())

        # 1. Extract text based on file type
        if uploaded_resume_file.name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(temp_resume_path)
        elif uploaded_resume_file.name.endswith('.docx'):
            resume_text = extract_text_from_docx(temp_resume_path)

        # Clean up the temporary file
        if os.path.exists(temp_resume_path):
            os.remove(temp_resume_path)

        if not resume_text:
            st.error("❌ Could not extract text from your resume. Please try a different file or format, or ensure it's not an image-only PDF.")
            st.stop() # Stop if text extraction failed

        # 2. Preprocess text for similarity calculation (remove noise, lemmatize)
        processed_resume_for_sim = preprocess_text_spacy(resume_text)
        processed_job_desc_for_sim = preprocess_text_spacy(job_description)

        # 3. Calculate Cosine Similarity Match Score
        similarity_score = calculate_cosine_similarity(processed_resume_for_sim, processed_job_desc_for_sim)
        match_percentage = round(similarity_score * 100, 2) # Convert to percentage and round

        st.subheader("📊 Your Resume Match Score")
        # Display score with a visual indicator (emoji)
        if match_percentage >= 75:
            st.success(f"🎉 **Excellent Match! Your resume matches {match_percentage}% of the job description.**")
            st.write("You seem to be a fantastic fit for this role!")
        elif match_percentage >= 50:
            st.info(f"👍 **Good Match! Your resume matches {match_percentage}% of the job description.**")
            st.write("You're on the right track! A few tweaks could make it even better.")
        else:
            st.warning(f"🤔 **Moderate Match. Your resume matches {match_percentage}% of the job description.**")
            st.write("There's potential, but consider tailoring your resume more closely to the job requirements.")

        # 4. Generate AI-powered feedback using Ollama
        st.subheader("✨ AI-Powered Tailoring Suggestions (from Ollama)")

        # Create a detailed prompt for Ollama, asking for specific types of feedback.
        # We limit the text length to avoid exceeding the AI model's context window.
        # This prompt is updated to be more directive to reduce generic suggestions.
        ollama_prompt = f"""
        You are an expert career advisor and an AI specialized in resume optimization.
        Your task is to analyze a given resume and a job description.
        Provide constructive, actionable feedback in a clear, well-structured format (using bullet points and bolding for readability).

        **CRITICAL INSTRUCTIONS FOR ACCURACY:**
        * **DO NOT** suggest adding sections (e.g., "Skills" or "Experience") if they are clearly discernable or implied as present in the provided resume text. Instead, focus on how to *improve* or *align* existing sections.
        * **DO NOT** give advice about elements *not present* in the resume text provided (e.g., a cover letter, references, etc.).
        * **PRIORITIZE** specific advice that directly addresses the content of *this* resume in relation to *this* job description, rather than generic resume tips.
        * **ENSURE** suggestions for quantifying results are framed as *revising existing bullet points* to include metrics, rather than implying a complete absence of accomplishments.

        Here's what I need you to focus on:

        1.  **Strengths (What Matches Well):**
            * Identify specific skills, experiences, or keywords from the resume that directly align with the job description.
            * Explain *why* these are strong matches, citing examples from the resume if possible.

        2.  **Areas for Improvement (What's Missing or Weakly Highlighted):**
            * Point out key skills, experiences, or keywords explicitly mentioned in the job description that are either genuinely absent from the resume or not clearly highlighted.
            * Be specific about what's missing (e.g., "missing explicit experience with AWS cloud," "lack of detail on project management methodologies like Agile/Scrum").

        3.  **Tailoring Suggestions (How to Enhance Existing Content):**
            * Provide concrete, actionable advice on how to modify *existing* resume content to better align with the job description.
            * Suggest specific phrases, achievements, or project types to emphasize.
            * Recommend rephrasing existing bullet points to better align with the job's language and requirements.
            * If a "Skills" section exists but isn't optimized, suggest how to make it more relevant (e.g., "Expand the 'Skills' section to prominently feature X, Y, Z from the job description," or "Group related skills under categories mentioned in the job description").

        4.  **Overall Fit (Summary):**
            * Give a brief, clear, and actionable overall assessment of how well the resume aligns with the job description, summarizing the main takeaways.

        ---
        **Candidate Resume (first 4000 characters for context, actual resume might be longer):**
        {resume_text[:4000]}
        ---
        **Job Description (first 4000 characters for context, actual JD might be longer):**
        {job_description[:4000]}
        ---

        Please present your feedback professionally and concisely using markdown formatting (like bullet points, bolding).
        """

        # Call the Ollama analysis function
        ollama_feedback = analyze_with_ollama(ollama_prompt)

        if ollama_feedback:
            st.markdown(ollama_feedback) # Display the AI's feedback
        else:
            st.error("Could not retrieve AI feedback. Please ensure Ollama is running and the model is downloaded correctly.")

