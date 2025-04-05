A Streamlit application for uploading, processing, and querying stock reports using a Retrieval-Augmented Generation (RAG) system.

## Features

- Upload PDF stock reports
- Extract text and metadata from PDFs
- Process text into chunks and generate embeddings
- Store documents and embeddings in a Lancedb database
- Query the system with natural language questions
- View relevant source information with highlighted matches
- System statistics and configuration information

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

### 1. Upload & Process
- Upload one or more PDF stock reports
- Process the documents to extract text and generate embeddings
- Wait for processing to complete

### 2. Query & Analysis
- Enter a natural language question about the stock reports
- Configure query parameters (sources to show, max results)
- View the answer and supporting sources

### 3. System Information
- View database statistics
- See system configuration
- Reset the database if needed
