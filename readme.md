# Stock Market Analysis System

A comprehensive application for analyzing stock market reports using advanced AI techniques. This system combines OCR, vector embeddings, and large language models to extract insights from financial PDFs through natural language queries.

## Problem Statement

Analyzing stock market reports manually is time-consuming and may miss important connections between data points across multiple reports. This solution automates the extraction, processing, and querying of financial information from PDF reports.

## Features

- Upload multiple PDF stock reports simultaneously
- Extract text using PyMuPDF with Tesseract OCR fallback for image-based content
- Process text into optimized chunks with metadata preservation
- Generate high-quality embeddings using Amazon Titan model
- Store documents and embeddings in a LanceDB vector database
- Query the system with natural language questions
- View AI-generated answers with supporting evidence from source documents
- Monitor system statistics and configuration

## Tech Stack

### Frontend
- **Streamlit**: Web application framework for the user interface

### Backend & Data Processing
- **Python**: Core programming language
- **PyMuPDF (fitz)**: Primary PDF text extraction
- **Tesseract OCR**: For image-based text extraction in PDFs
- **LanceDB**: Vector database for storing document chunks and embeddings
- **Pandas/NumPy**: For data manipulation and processing

### AI & Machine Learning
- **AWS Bedrock**: For accessing Claude AI model
- **Amazon Titan Embeddings**: For generating vector embeddings
- **Concurrent Processing**: Multithreaded batch processing

### Cloud & Infrastructure
- **AWS**: For AI model access through Bedrock
- **Boto3**: AWS SDK for Python

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env` file:
```
AWS_REGION=your_aws_region
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
INFERENCE_PROFILE_ARN=your_inference_profile_arn
```

4. Install Tesseract OCR (optional, for improved text extraction):
   - Windows: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

5. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

### 1. Upload & Process
- Upload one or more PDF stock reports
- Click "Process Reports" to extract text and generate embeddings
- Monitor progress as the system processes each document

### 2. Query & Analysis
- Enter a natural language question about the stock reports
- Configure query parameters (sources to show, max results)
- View the AI-generated answer and supporting sources with page references
- Explore highlighted text matches from the original documents

### 3. System Information
- View database statistics including total documents, chunks, and embedding dimensions
- See system configuration settings
- Reset the database if needed

## Advanced Features

- **Concurrent Processing**: Efficiently handles large documents with multithreading
- **Caching System**: Prevents redundant processing of previously analyzed documents
- **Error Handling**: Robust fallback mechanisms for text extraction
- **Result Reranking**: Intelligent sorting of search results based on relevance

## Project Structure

- `app.py`: Main Streamlit application
- `extract_pdf.py`: PDF processing and text extraction
- `embeddings.py`: Vector embedding generation and database management
- `rag_system.py`: Retrieval-Augmented Generation system implementation

## License

© 2025 Stock Report Analysis Tool
