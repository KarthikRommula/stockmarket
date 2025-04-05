import os
import json
import boto3
import lancedb
import pandas as pd
import numpy as np
import time
import random
from typing import Dict, Any
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from embeddings import generate_embedding

# Load environment variables
load_dotenv()

# Configure AWS Bedrock client for embeddings (if needed)
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

# Using AWS Bedrock for Claude model access
print("Using AWS Bedrock for Claude model access")

# Get inference profile ID from environment variables or use a default value
# You'll need to set this in your .env file
INFERENCE_PROFILE_ID = os.getenv('INFERENCE_PROFILE_ARN')

def exponential_backoff(attempt, max_attempts=5, base_delay=1, max_delay=60):
    """
    Calculate delay for exponential backoff with jitter.
    
    Args:
        attempt: Current attempt number (zero-based)
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        float: Delay time in seconds, or None if max attempts reached
    """
    if attempt >= max_attempts:
        return None
        
    # Calculate exponential backoff with full jitter
    delay = min(max_delay, base_delay * (2 ** attempt))
    jitter = random.uniform(0, delay)
    return jitter

class StockReportRAG:
    def __init__(self, db_path="stockreports_db"):
        """Initialize the RAG system"""
        # Connect to LanceDB
        self.db = lancedb.connect(db_path)
        
        if "stock_chunks" not in self.db.table_names():
            raise ValueError("Database not initialized. Please run embeddings.py first.")
        
        self.table = self.db.open_table("stock_chunks")
        print(f"Connected to database with {len(self.table)} chunks")
    
    def retrieve(self, query, top_k=5):
        """Retrieve relevant passages based on query"""
        # Generate embedding for query
        query_embedding = generate_embedding(query)
        
        if not query_embedding:
            print("Error generating query embedding")
            return pd.DataFrame()
        
        # Search for similar chunks
        try:
            results = self.table.search(query_embedding).limit(top_k).to_pandas()
            
            # FIX: Convert any non-numeric distance/similarity metrics to string
            # This is to prevent Arrow conversion errors when displaying in Streamlit
            if '_distance' in results.columns and results['_distance'].dtype == 'object':
                results['_distance'] = results['_distance'].apply(
                    lambda x: float(x) if isinstance(x, (int, float, np.number)) else 0.5
                )
                
            # Fix for metric type columns that might contain 'cosine'
            for col in results.columns:
                if col.endswith('_distance') or col.endswith('_similarity') or col.endswith('_metric'):
                    if results[col].dtype == 'object':
                        results[col] = results[col].astype(str)
                
            return results
        except Exception as e:
            print(f"Error in retrieve: {e}")
            return pd.DataFrame()
    
    def rerank_results(self, results, query):
        """Simple reranking heuristic based on page numbers and text length"""
        # This is a basic reranking that could be improved
        # Factors we consider:
        # 1. Original similarity score
        # 2. Lower page numbers might be more important (executive summary, etc.)
        # 3. Longer text chunks might contain more information
        
        if len(results) <= 1:
            return results
        
        # Calculate min pages for each result (first page of each chunk)
        results['min_page'] = results['pages'].apply(lambda x: min(x) if len(x) > 0 else 999)
        
        # Calculate text length
        results['text_length'] = results['text'].apply(len)
        
        # Normalize values between 0 and 1
        min_page_min = results['min_page'].min()
        min_page_max = max(results['min_page'].max(), min_page_min + 1)  # Avoid division by zero
        
        text_len_min = results['text_length'].min()
        text_len_max = max(results['text_length'].max(), text_len_min + 1)  # Avoid division by zero
        
        # Calculate page factor (lower pages get higher scores)
        results['page_factor'] = 1 - ((results['min_page'] - min_page_min) / (min_page_max - min_page_min))
        
        # Calculate length factor (longer texts get higher scores, but with diminishing returns)
        results['length_factor'] = (results['text_length'] - text_len_min) / (text_len_max - text_len_min)
        
        # Calculate combined score
        # We use the original _distance as a primary factor, and then adjust with the other factors
        # Lower _distance means higher similarity
        # FIX: Make sure _distance is numeric for calculations
        if '_distance' in results.columns:
            if results['_distance'].dtype == 'object':
                # Convert to numeric, coercing errors to NaN, then fill NaN with a default value
                results['_distance'] = pd.to_numeric(results['_distance'], errors='coerce').fillna(0.5)
            
            results['adjusted_score'] = (1 - results['_distance']) * 0.7 + results['page_factor'] * 0.2 + results['length_factor'] * 0.1
        else:
            # If _distance doesn't exist, just use the other factors
            results['adjusted_score'] = results['page_factor'] * 0.6 + results['length_factor'] * 0.4
        
        # Sort by the new score
        results = results.sort_values('adjusted_score', ascending=False)
        
        return results

    def format_context(self, results):
        """Format retrieval results into a context string for Claude"""
        context_parts = []
        
        for i, row in results.iterrows():
            source = os.path.basename(row['source_file']).replace('.json', '.pdf')
            
            # FIX: Ensure pages is properly formatted
            if isinstance(row['pages'], (list, np.ndarray)) and len(row['pages']) > 0:
                pages = [str(p) for p in row['pages']]
                pages_str = ", ".join(pages)
            else:
                pages_str = "N/A"
            
            context_part = f"[DOCUMENT {i+1}] From '{source}', Pages {pages_str}:\n{row['text']}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

    def generate_response(self, query, context):
        """Generate a response using Claude based on the query and retrieved context"""
        system_prompt = "You are a specialized stock analysis assistant that helps analyze financial reports."
        
        user_prompt = f"""I'll provide you with context from various stock reports, and you need to answer the question based only on this information.

CONTEXT:
{context}

USER QUESTION:
{query}

Please provide a comprehensive answer based only on the information in the context. If the context doesn't contain the answer, say that you don't have enough information rather than speculating. Include relevant financial figures where available."""

        max_attempts = 5
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Use AWS Bedrock for Claude 3.5 Sonnet
                if INFERENCE_PROFILE_ID:
                    # Use the profile ID directly as the model ID
                    print(f"Using Claude 3.5 Sonnet model: {INFERENCE_PROFILE_ID}")
                    response = bedrock_runtime.invoke_model(
                        modelId=INFERENCE_PROFILE_ID,
                        body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1000,
                        "messages": [
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        "system": system_prompt
                    })
                )
                
                    response_body = json.loads(response['body'].read())
                    answer = response_body['content'][0]['text']
                    
                    return answer
                else:
                    # Fallback to Claude 3 Sonnet if no inference profile is set
                    print("No inference profile found. Falling back to Claude 3 Sonnet.")
                    print("Using Claude 3 Sonnet model: anthropic.claude-3-sonnet-20240229-v1:0")
                    response = bedrock_runtime.invoke_model(
                        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 1000,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_prompt
                                }
                            ],
                            "system": system_prompt
                        })
                    )
                    
                    response_body = json.loads(response['body'].read())
                    answer = response_body['content'][0]['text']
                    
                    return answer
                
            except Exception as e:
                # Handle rate limiting and other errors
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    delay = exponential_backoff(attempt, max_attempts)
                    if delay is None:
                        print(f"Maximum retry attempts ({max_attempts}) reached.")
                        return f"Error generating response: Maximum retry attempts reached due to rate limiting. Please try again later."
                    
                    print(f"Rate limited, retrying in {delay:.2f} seconds (attempt {attempt+1}/{max_attempts})...")
                    time.sleep(delay)
                    attempt += 1
                else:
                    print(f"Anthropic API error: {e}")
                    return f"Error generating response: {str(e)}"
            
        return "Error: Maximum retry attempts reached. Please try again later."

    def process_query(self, query, top_k=5):
        """Process a query through the complete RAG pipeline"""
        print(f"Processing query: '{query}'")
        
        # Retrieve relevant passages
        print("Retrieving relevant passages...")
        results = self.retrieve(query, top_k)
        
        if results.empty:
            return "No relevant information found in the database."
        
        # Rerank results
        print("Reranking results...")
        reranked_results = self.rerank_results(results, query)
        
        # Format context
        context = self.format_context(reranked_results)
        
        # Generate response
        print("Generating response...")
        response = self.generate_response(query, context)
        
        return response

    def get_source_information(self):
        """Get information about available sources in the database"""
        # FIX: Handle potential Arrow conversion errors
        try:
            df = self.table.to_pandas()
            
            # Ensure all columns are of compatible types for Arrow conversion
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'text' and col != 'summary' and col != 'source_file' and col != 'chunk_id':
                    # Convert non-string object columns to string
                    df[col] = df[col].astype(str)
            
            unique_sources = df['source_file'].unique()
            
            sources_info = []
            for source in unique_sources:
                source_chunks = df[df['source_file'] == source]
                source_info = {
                    "source": os.path.basename(source),
                    "chunks": len(source_chunks),
                    # Fixed the boolean context issue with NumPy arrays
                    "pages": sorted(set([p for pages in source_chunks['pages'] for p in pages if isinstance(pages, (list, np.ndarray)) and len(pages) > 0]))
                }
                sources_info.append(source_info)
            
            return sources_info
        except Exception as e:
            print(f"Error in get_source_information: {e}")
            return []
    
    def get_system_stats(self):
        """
        Get statistics about the RAG system.
        
        Returns:
            dict: Statistics including total documents, chunks, and embedding dimensions
        """
        try:
            # Get DataFrame from LanceDB table
            df = self.table.to_pandas()
            
            # Get total unique documents
            total_documents = df['source_file'].nunique()
            
            # Get total chunks
            total_chunks = len(df)
            
            # Get embedding dimensions (if there's data)
            embedding_dimensions = 0
            if not df.empty and 'embedding' in df.columns:
                # Get the first non-null embedding
                for i in range(len(df)):
                    embedding = df.iloc[i]['embedding']
                    if embedding is not None and len(embedding) > 0:
                        embedding_dimensions = len(embedding)
                        break
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "embedding_dimensions": embedding_dimensions,
                "similarity_metric": "cosine"  # FIX: Add this as string instead of in DataFrame
            }
        except Exception as e:
            print(f"Error in get_system_stats: {e}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "embedding_dimensions": 0,
                "similarity_metric": "cosine"
            }
    
    def get_config(self):
        """
        Get the configuration of the RAG system.
        
        Returns:
            dict: Configuration settings
        """
        # FIX: Ensure all values are of compatible types for Arrow conversion
        return {
            "chunk_size": 1000,  # Assuming default chunk size of 1000
            "chunk_overlap": 200,  # Assuming default overlap of 200
            "similarity_metric": "cosine",  # Use string instead of object
            "model": "Amazon Bedrock embedding model"  # Update with your actual model
        }
    
    def query(self, query_text, max_sources=3):
        """
        Query the RAG system with a question (wrapper for process_query to match app.py interface).
        
        Args:
            query_text (str): The question to ask
            max_sources (int): Maximum number of sources to retrieve
            
        Returns:
            dict: Response containing answer and sources
        """
        # Retrieve and rerank results
        results = self.retrieve(query_text, max_sources)
        
        if results.empty:
            return {
                "answer": "No relevant information found in the database.",
                "sources": []
            }
        
        reranked_results = self.rerank_results(results, query_text)
        
        # Format context and generate response
        context = self.format_context(reranked_results)
        answer = self.generate_response(query_text, context)
        
        # Format sources in the expected format for app.py
        sources = []
        for i, row in reranked_results.iterrows():
            source = os.path.basename(row['source_file']).replace('.json', '.pdf')
            
            # FIX: Handle page number formatting properly
            if isinstance(row['pages'], (list, np.ndarray)) and len(row['pages']) > 0:
                page = str(min(row['pages']))
            else:
                page = "N/A"
            
            # FIX: Make sure we have a numeric score
            if '_distance' in row and (isinstance(row['_distance'], (int, float, np.number)) or 
                                     (isinstance(row['_distance'], str) and row['_distance'].replace('.', '', 1).isdigit())):
                try:
                    score = float(1 - float(row['_distance']))
                except (ValueError, TypeError):
                    score = 0.5
            else:
                score = 0.5
            
            # FIX: Create properly formatted source info
            sources.append({
                "source": source,
                "page": page,  # Now properly formatted as a string
                "text": row['text'],
                "score": score
            })
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def reset_database(self):
        """
        Reset the database by dropping the table.
        """
        try:
            if "stock_chunks" in self.db.table_names():
                self.db.drop_table("stock_chunks")
            print("Database has been reset.")
        except Exception as e:
            print(f"Error resetting database: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rag_system.py <query>")
        sys.exit(1)
    
    query = sys.argv[1]
    
    rag = StockReportRAG()
    response = rag.process_query(query)
    
    print("\n--- RESPONSE ---")
    print(response)