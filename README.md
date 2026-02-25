💬 RAG Chatbot – Knowledge Base Assistant
A Retrieval‑Augmented Generation (RAG) chatbot that answers user questions strictly based on indexed documents.
Built using Azure OpenAI, Azure AI Search, and Streamlit.

This demo is grounded on the Swiggy Annual Report FY 2023–24.

🧠 Architecture Overview
Frontend: Streamlit chat UI
LLM: Azure OpenAI (GPT‑4o)
Vector Store: Azure AI Search
Embeddings: Azure OpenAI text embeddings
Data Source: PDF documents
📄 PDF Ingestion (How It Works)
PDF ingestion is handled in ingest_pdf.py and is a one‑time setup per document.

Ingestion Steps
PDF → Images

Uses pdf2image (Poppler backend)
Each page rendered at 300 DPI
Vision‑Based Extraction

Each page image is sent to GPT‑4o Vision
Extracts all visible content:
Financial figures (exact values)
Tables (converted to markdown)
Charts and trends
Text content
Page Summarization

GPT‑4o summarizes each page into 3–5 key bullet points
Focused on important financial and operational insights
Embedding Creation

Extracted content and summaries are embedded using Azure OpenAI embedding model
Enables semantic similarity search
Indexing in Azure AI Search

Each page is stored as searchable documents with vectors
Key fields:
content
contentVector
doc
page
content_type
🔍 Question Answering (RAG Flow)
User enters a question in the Streamlit UI
Query is embedded using Azure OpenAI
Azure AI Search retrieves the most relevant content (vector search)
Retrieved context is passed to GPT‑4o
GPT‑4o generates a grounded answer only from retrieved documents
✅ No hallucinations
✅ No external knowledge

🏗️ Project Structure
Rag_Chatbot/
├── app.py            # Streamlit chatbot UI
├── ingest_pdf.py     # PDF ingestion & indexing
├── config.py         # Azure configuration
├── requirements.txt  # Dependencies
├── files/            # Source PDFs
└── README.md

🚀 Running the Project
Install Dependencies
pip install -r requirements.txt

Ingest PDF (One‑Time)
python ingest_pdf.py

Run Chatbot
streamlit run app.py

🧰 Technologies Used
Azure OpenAI (GPT‑4o) – Vision extraction & response generation
Azure AI Search – Vector storage and retrieval
Streamlit – Chat UI
pdf2image + Poppler – PDF rendering
✅ Key Features
Vision‑based PDF data extraction
Semantic vector search
Document‑grounded answers
Streaming chat interface
Suggested questions for demos
This project demonstrates an end‑to‑end, production‑ready RAG pipeline using Azure services.
