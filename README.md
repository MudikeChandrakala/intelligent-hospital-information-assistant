# 🏥 Intelligent Hospital Information Assistant

An AI-powered hospital information assistant built using **Retrieval-Augmented Generation (RAG)** and **Google Gemini**. The system provides hospital information through a text-based AI Assistant, Voice Assistant, Medical Report Analysis, and Prescription Analysis modules.

---

## 📌 Project Overview

The **Intelligent Hospital Information Assistant** is designed to provide users with fast and accessible hospital information through natural-language interaction.

The system combines:

- Hospital knowledge base
- Semantic document retrieval
- Hugging Face embeddings
- ChromaDB vector storage
- Google Gemini
- LangChain
- Streamlit
- Voice interaction
- Medical report analysis
- Prescription analysis

The assistant grounds its responses in the hospital information available in the project's knowledge base.

---

## ✨ Key Features

### 💬 1. AI Assistant

Users can ask natural-language questions about:

- Hospital departments
- Department locations
- Department services
- Doctors and their details
- Consultation information
- Appointments
- Hospital navigation
- Insurance
- Medicines
- Emergency procedures
- Frequently asked questions
- Patient guidelines
- Billing information

The AI Assistant uses a **Retrieval-Augmented Generation (RAG)** pipeline to retrieve relevant hospital information before generating a response.

---

### 🎤 2. Voice Assistant

The Voice Assistant provides hospital information through voice interaction.

Features include:

- Voice input
- Natural-language hospital queries
- Context-aware follow-up questions
- AI-generated responses
- Spoken responses

The Voice Assistant uses the same hospital knowledge and RAG backend while maintaining its own conversation history.

---

### 📋 3. Medical Report Analysis

The Medical Report Analysis module processes laboratory report information and provides a structured interpretation.

It can identify:

- Detected parameters
- Normal results
- High results
- Low results
- Unknown results
- Reference ranges
- Important findings
- Results requiring manual verification
- General informational guidance
- AI-generated report explanations

> **Disclaimer:** Medical report analysis is informational only and does not provide a medical diagnosis.

---

### 💊 4. Prescription Analysis

The project includes dedicated prescription analysis functionality for processing prescription information and generating AI-assisted interpretations.

The implementation includes:

- `prescription_analyzer.py`
- `prescription_ai_service.py`

---

## 🧠 RAG Architecture

The core AI Assistant follows a Retrieval-Augmented Generation architecture:

```text
                User Question
                     │
                     ▼
             Question Processing
                     │
                     ▼
            Embedding Generation
                     │
                     ▼
              ChromaDB Search
                     │
                     ▼
          Relevant Hospital Documents
                     │
                     ▼
               Prompt Builder
                     │
                     ▼
              Google Gemini
                     │
                     ▼
             Assistant Response

Research Project/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── knowledge_base/
│   │
│   ├── structured/
│   │   ├── appointments.json
│   │   ├── department_master.json
│   │   ├── disease_mapping.json
│   │   ├── doctor_directory.json
│   │   ├── emergency_protocols.json
│   │   ├── insurance.json
│   │   ├── medicine_database.json
│   │   ├── navigation.json
│   │   └── symptom_mapping.json
│   │
│   └── unstructured/
│       ├── billing_information.txt
│       ├── faq.txt
│       ├── hospital_information.txt
│       └── patient_guidelines.txt
│
├── modules/
│   ├── chroma_vector_store.py
│   ├── document_loader.py
│   ├── embedding_generator.py
│   ├── gemini_client.py
│   ├── prescription_ai_service.py
│   ├── prescription_analyzer.py
│   ├── prompt_builder.py
│   ├── rag_pipeline.py
│   ├── reference_ranges.py
│   ├── report_ai_service.py
│   ├── report_analyzer.py
│   ├── retriever.py
│   ├── text_chunker.py
│   ├── voice_assistant.py
│   └── __init__.py
│
├── ui/
│   ├── chat.py
│   ├── components.py
│   ├── layout.py
│   ├── metrics.py
│   ├── report_analysis.py
│   ├── sidebar.py
│   ├── styles.py
│   └── utils.py
│
└── tests/
    ├── test_document_loader.py
    ├── test_gemini_client.py
    ├── test_rag_pipeline.py
    ├── test_reference_range_ocr_improvements.py
    ├── test_reference_ranges.py
    ├── test_report_ai_service.py
    ├── test_report_analysis_presentation.py
    └── test_report_analyzer.py

Technologies Used
Component	Technology
Programming Language	Python 3.10.x
User Interface	Streamlit
Large Language Model	Google Gemini
RAG Framework	LangChain
Embeddings	Hugging Face / Sentence Transformers
Vector Database	ChromaDB
Environment Management	python-dotenv
Report Generation	ReportLab
Testing Framework	pytest
📦 Installation
1. Clone the Repository
git clone <repository-url>
cd "Research Project"
2. Create a Virtual Environment
python -m venv .venv
3. Activate the Virtual Environment

For Windows PowerShell:

.\.venv\Scripts\Activate.ps1
4. Install Dependencies
pip install -r requirements.txt
🔐 Environment Configuration

Create a .env file in the project root.

Add your Google Gemini API key:

GEMINI_API_KEY=your_api_key_here

Never commit .env or expose your API key publicly.

The .env file should remain excluded through .gitignore.

▶️ Running the Application

Activate the virtual environment:

.\.venv\Scripts\Activate.ps1

Start the Streamlit application:

streamlit run app.py

The application will normally be available at:

http://localhost:8501
🧪 Testing

The project contains automated tests covering the major application components.

Run the complete test suite:

.\.venv\Scripts\python.exe -m pytest -q
Current Test Status

The complete automated test suite has been successfully validated with:

159 passed

This confirms that the current automated test suite passes successfully.

📚 Knowledge Base

The hospital knowledge base consists of structured and unstructured information.

Structured Knowledge

The structured JSON datasets contain information about:

Departments
Doctors
Symptoms
Diseases
Medicines
Appointments
Insurance
Navigation
Emergency protocols
Unstructured Knowledge

The unstructured knowledge base contains:

Hospital information
FAQs
Patient guidelines
Billing information

These sources are loaded, processed, embedded, and made available for semantic retrieval.

🔄 Conversation Handling

The application supports context-aware conversations.

The AI Assistant and Voice Assistant maintain separate conversation histories.

For example:

User:
My father has chest pain. Which department should I visit?

Assistant:
Please provide your father's age.

User:
43

Assistant:
Uses the previous question and the provided age
to continue the original request.

This allows short follow-up responses, such as an age, to be interpreted in the context of the preceding conversation.

The system also distinguishes between:

Direct department questions
First-person symptom questions
Family-member symptom questions
Doctor recommendation requests
Emergency-related questions
🏥 Medical Information Safety

The assistant is intended to provide hospital information and informational support.

AI-generated medical explanations should not be treated as a diagnosis or a substitute for professional medical advice.

For emergencies, users should follow the hospital's emergency procedures and seek immediate professional medical assistance.

Medical report analysis is provided for informational purposes and should be reviewed by a qualified healthcare professional when appropriate.

📊 System Modules
Module	Responsibility
document_loader.py	Loads hospital knowledge-base documents
embedding_generator.py	Generates document embeddings
chroma_vector_store.py	Manages vector storage
retriever.py	Retrieves relevant hospital documents
prompt_builder.py	Builds grounded RAG prompts
gemini_client.py	Communicates with Google Gemini
rag_pipeline.py	Coordinates retrieval and response generation
voice_assistant.py	Supports voice functionality
report_analyzer.py	Analyzes medical reports
report_ai_service.py	Generates AI report explanations
prescription_analyzer.py	Processes prescriptions
prescription_ai_service.py	Provides prescription AI functionality
reference_ranges.py	Handles laboratory reference ranges
🎯 Project Objectives

The main objectives of the project are to:

Provide accessible hospital information through natural language.
Reduce the effort required to search hospital information manually.
Provide context-aware conversational interaction.
Support voice-based hospital information access.
Analyze medical laboratory reports in an understandable format.
Use RAG to ground AI responses in hospital-specific information.
Provide a modular and testable application architecture.
Provide a user-friendly interface for hospital information access.
🔍 Example Use Cases
Hospital Information
Where is the Neurology department located?
Department Recommendation
Which department should a patient with chest pain visit?
Doctor Recommendation
I have chest pain. Which doctor should I consult?
Context-Aware Follow-up
My father has chest pain. Which department should I visit?

43
Emergency Information
What should I do if I have chest pain in an emergency?
Medical Report Analysis

Users can upload or provide medical laboratory report information for structured analysis and AI-assisted explanation.

🚀 Project Status

Final Development Stage

Core functionality has been implemented and the automated test suite has been validated with:

159 passed

The final live AI and Voice Assistant validation is performed when the Google Gemini usage quota is available.

After final validation, the project will undergo the final Git review and commit.