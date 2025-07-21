# 🤖 AI Resume Analyzer & Job Match Platform

## Project Overview

This is an AI-powered platform designed to streamline the resume analysis and job matching process. It helps job seekers tailor their resumes to specific job descriptions and assists recruiters in quickly identifying the most relevant candidates. The platform leverages local AI models (via Ollama) and Natural Language Processing (NLP) techniques for intelligent feedback and similarity calculations.

---

## ✨ Features

* **Intelligent Resume Analysis:** Upload your resume (PDF or DOCX) for detailed content extraction.
* **Job Description Input:** Easily paste or type in the job description you're targeting.
* **Cosine Similarity Matching:** Get an accurate percentage score indicating how well your resume matches the job description based on content relevance.
* **AI-Powered Feedback:** Receive tailored suggestions from a local AI model (Gemma from Ollama) on:
    * Your resume's strengths relevant to the job.
    * Key areas for improvement (missing skills, experiences, or keywords).
    * Specific recommendations to enhance your resume's alignment.
    * An overall assessment of fit.
* **User-Friendly Interface:** Built with Streamlit for an intuitive and interactive web application experience.
* **Local AI Processing:** Utilizes Ollama to run large language models directly on your machine, ensuring data privacy and potentially faster processing without relying on cloud APIs for core AI analysis.

---

## 🚀 How to Run the Project Locally

Follow these steps to set up and run the AI Resume Analyzer on your computer.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.11.7** (or compatible 3.x version)
    * Download from: [python.org](https://www.python.org/downloads/)
    * **Important (Windows):** During installation, make sure to check "Add python.exe to PATH".
* **Visual Studio Code (VS Code)**
    * Download from: [code.visualstudio.com](https://code.visualstudio.com/)
* **Ollama**
    * Download and install the Ollama server for your OS: [ollama.com/download](https://ollama.com/download)
    * After installation, verify it's running in your system tray (Windows) or as a background process.

### Setup Steps

1.  **Clone the Repository (or navigate to your project folder):**
    Open your terminal/command prompt and run:
    ```bash
    git clone [https://github.com/Saivarshini-001/AI-Resume-Analyzer.git](https://github.com/Saivarshini-001/AI-Resume-Analyzer.git)
    cd AI-Resume-Analyzer
    ```
    If you've followed the guide above, you are already in the `AI-Resume-Analyzer` folder.

2.  **Create and Activate a Python Virtual Environment:**
    This isolates your project's dependencies.
    * Open the integrated terminal in VS Code (`Ctrl+` or `Cmd+`).
    * Create the virtual environment:
        ```bash
        python -m venv venv
        ```
    * Activate the virtual environment:
        * **Windows:**
            ```bash
            .\venv\Scripts\activate
            ```
        * **macOS / Linux:**
            ```bash
            source venv/bin/activate
            ```
    * You should see `(venv)` at the start of your terminal prompt.

3.  **Install Project Dependencies:**
    With your virtual environment activated, install the required Python libraries listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download spaCy Language Model:**
    The `en_core_web_sm` model is needed for basic text processing.
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Download the Ollama AI Model:**
    You need to download the `gemma:2b` model which is used for AI feedback.
    * Open a **new** terminal or command prompt (this one doesn't need to have your `(venv)` active, as `ollama` is a system command).
    * Run the command to pull the model:
        ```bash
        ollama run gemma:2b
        ```
        * Wait for the download to complete (it will show `success` and then prompt you to send a message; you can type `/bye` to exit the interactive session).

6.  **Run the Streamlit Application:**
    Ensure your `(venv)` is still active in your VS Code terminal.
    ```bash
    python -m streamlit run app.py
    ```
    This command will open the application in your default web browser.

---
## 🙏 Acknowledgments

* Streamlit for the fantastic web framework.
* Ollama for enabling local LLM inference.
* spaCy for robust NLP capabilities.
* The open-source community for amazing libraries!
