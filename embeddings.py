import os
import json
import boto3
import lancedb
import numpy as np
import pyarrow as pa
import time
import random
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure AWS Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def generate_embedding(text, max_retries=8):
    """Generate embedding using Titan Text Embeddings model on AWS Bedrock with enhanced retry logic"""
    retries = 0
    while retries <= max_retries:
        try:
            response = bedrock_runtime.invoke_model(
                modelId='amazon.titan-embed-text-v2:0',
                body=json.dumps({
                    "inputText": text
                })
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding', [])
            
            # Convert to numpy array for easier manipulation
            embedding_np = np.array(embedding, dtype=np.float32)
            
            # Ensure the embedding is exactly 1536 dimensions
            if len(embedding_np) != 1536:
                print(f"Warning: Got embedding of size {len(embedding_np)}, padding/truncating to 1536")
                if len(embedding_np) < 1536:
                    # Pad with zeros if too short
                    embedding_np = np.pad(embedding_np, (0, 1536 - len(embedding_np)), 'constant')
                else:
                    # Truncate if too long
                    embedding_np = embedding_np[:1536]
                    
            return embedding_np.tolist()
        
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                retries += 1
                if retries <= max_retries:
                    # Enhanced exponential backoff with jitter
                    sleep_time = (3 ** retries) + random.random() * 2
                    print(f"Throttling error. Retrying in {sleep_time:.2f} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    print(f"Max retries reached after throttling. Using zero vector.")
                    return [0.0] * 1536
            else:
                print(f"Error generating embedding: {e}")
                return [0.0] * 1536
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Create a zero vector of exactly 1536 dimensions
            return [0.0] * 1536

def setup_lancedb(db_path="stockreports_db"):
    """Set up LanceDB database"""
    # Initialize LanceDB
    db = lancedb.connect(db_path)
    
    # Create table if it doesn't exist
    if "stock_chunks" not in db.table_names():
        # Create a proper PyArrow schema with FIXED size list for embeddings
        schema = pa.schema([
            pa.field("text", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("pages", pa.list_(pa.int32())),
            pa.field("embedding", pa.list_(pa.float32(), 1536)),  # Fixed-length list of float32
            pa.field("chunk_id", pa.string())
        ])
        
        db.create_table("stock_chunks", schema=schema)
    
    return db.open_table("stock_chunks")

def process_extractions_to_embeddings(extracted_dir="extracted_data", db_path="stockreports_db"):
    """Process all extracted JSON files and generate embeddings"""
    # Setup LanceDB
    table = setup_lancedb(db_path)
    
    # Get all JSON files in the extracted directory
    json_files = [os.path.join(extracted_dir, f) for f in os.listdir(extracted_dir) 
                 if f.endswith('.json')]
    
    if not json_files:
        print(f"No extracted data files found in {extracted_dir}")
        return table
    
    print(f"Found {len(json_files)} extracted data files")
    
    # Process each file
    for file_idx, json_file in enumerate(json_files):
        print(f"\nProcessing {os.path.basename(json_file)}... ({file_idx+1}/{len(json_files)})")
        
        # Load the extracted data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        source_file = data.get("source", os.path.basename(json_file))
        chunks = data.get("chunks", [])
        
        print(f"Found {len(chunks)} chunks to process")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            summary = chunk.get("summary", "")
            metadata = chunk.get("metadata", {})
            
            # Extract and process pages - improved handling
            pages = metadata.get("pages", [])
            print(f"Chunk {i} raw pages data: {pages}")
            
            # Ensure pages is a list of integers
            if isinstance(pages, list):
                # Clean and convert page numbers to integers
                cleaned_pages = []
                for p in pages:
                    try:
                        # Convert various page number formats to int
                        if isinstance(p, (int, float)):
                            cleaned_pages.append(int(p))
                        elif isinstance(p, str) and p.strip().isdigit():
                            cleaned_pages.append(int(p.strip()))
                    except (ValueError, TypeError):
                        pass  # Skip invalid values
                
                pages = cleaned_pages
                print(f"Chunk {i} cleaned pages: {pages}")
            else:
                print(f"Chunk {i} has invalid pages format: {type(pages)}")
                pages = []
            
            # Generate unique ID for this chunk
            chunk_id = f"{os.path.basename(source_file).replace('.json', '')}_{i}"
            
            # Generate embedding
            print(f"Generating embedding for chunk {i+1}/{len(chunks)}...")
            embedding = generate_embedding(text)
            
            if embedding:
                # Make sure embedding is a list of floats with exactly 1536 dimensions
                embedding = np.array(embedding, dtype=np.float32).tolist()
                
                # Prepare record for LanceDB
                record = {
                    "text": text,
                    "summary": summary,
                    "source_file": source_file,
                    "pages": pages,  # Use the cleaned pages list
                    "embedding": embedding,
                    "chunk_id": chunk_id
                }
                
                # Add to LanceDB
                table.add([record])
                print(f"Added chunk {i+1} to database with pages {pages}")
            else:
                print(f"Skipping chunk {i+1} due to embedding generation error")
                
            # Increased pause time between embeddings to avoid throttling
            if i > 0 and i % 2 == 0:  # Every 2 chunks
                delay = 5  # 5 seconds
                print(f"Pausing for {delay} seconds to avoid throttling...")
                time.sleep(delay)
            
            # Additional random pause to break patterns
            if random.random() < 0.2:  # 20% chance
                rand_delay = random.uniform(1, 3)
                print(f"Adding random pause of {rand_delay:.2f} seconds...")
                time.sleep(rand_delay)
    
    # Create vector index for faster similarity search - Using simplified approach
    print("\nCreating vector index...")
    create_vector_index(table)

    print("\nEmbedding generation and storage complete!")
    return table

def create_vector_index(table):
    """Create vector index with proper distance metric - simplified approach"""
    # Try FTS index with replace=True since the error indicates it already exists
    try:
        if hasattr(table, 'create_fts_index'):
            table.create_fts_index(["text"], replace=True)
            print("Created FTS index with replace=True")
            return True
    except Exception as e:
        print(f"FTS index creation failed: {e}")
    
    # Try various vector index approaches with error handling
    methods = [
        # Method 1: Basic with replace=True
        lambda: table.create_index("embedding", replace=True),
        
        # Method 2: With explicitly named parameters
        lambda: table.create_index(vector_column="embedding", replace=True),
        
        # Method 3: With metric parameter
        lambda: table.create_index("embedding", metric="cosine", replace=True),
    ]
    
    for i, method in enumerate(methods):
        try:
            method()
            print(f"Vector index created with method {i+1}")
            return True
        except Exception as e:
            print(f"Method {i+1} failed: {e}")
    
    print("All index creation attempts failed. Continuing without index...")
    return False

def test_retrieval(query, table, top_k=3, max_retries=5):
    """Test basic retrieval function with improved page number handling"""
    # Generate embedding for query with retry
    for attempt in range(max_retries):
        try:
            print(f"Generating query embedding (attempt {attempt+1}/{max_retries})...")
            query_embedding = generate_embedding(query)
            
            if not query_embedding or all(v == 0 for v in query_embedding):
                print("Error generating query embedding, retrying...")
                time.sleep(5 + random.random() * 3)
                continue
                
            # Get the results
            try:
                # Newer versions
                results = table.search(query_embedding).limit(top_k).to_list()
            except AttributeError:
                try:
                    # Older versions might use to_pandas() instead
                    import pandas as pd
                    results = table.search(query_embedding).limit(top_k).to_pandas().to_dict('records')
                except Exception:
                    # Fall back to raw arrow conversion
                    results = [row for row in table.search(query_embedding).limit(top_k).to_arrow().to_pylist()]
                
            print(f"\nTop {len(results)} results:")
            for i, row in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                
                # Format source file
                source_file = row.get('source_file', 'Unknown')
                # If source is a full path, extract just the filename
                if os.path.sep in source_file:
                    source_file = os.path.basename(source_file)
                print(f"Source: {source_file}")
                
                # Process page numbers with improved handling
                pages = row.get('pages', [])
                if pages and isinstance(pages, list):
                    # Format page numbers
                    try:
                        page_numbers = []
                        for p in pages:
                            if p is not None:
                                if isinstance(p, (int, float)):
                                    page_numbers.append(str(int(p)))
                                elif isinstance(p, str) and p.strip().isdigit():
                                    page_numbers.append(p.strip())
                        
                        if page_numbers:
                            print(f"Pages: {', '.join(page_numbers)}")
                        else:
                            print("Pages: Not available")
                    except Exception as e:
                        print(f"Pages: Error formatting ({str(e)})")
                else:
                    print("Pages: Not available")
                
                # Summary or text excerpt
                summary = row.get('summary', '')
                if summary:
                    if isinstance(summary, str):
                        print(f"Summary: {summary[:300]}...")
                    else:
                        print(f"Summary: {str(summary)[:300]}...")
                else:
                    text = row.get('text', '')
                    if isinstance(text, str):
                        print(f"Text: {text[:300]}...")
                    else:
                        print(f"Text: {str(text)[:300]}...")
            
            return results
            
        except Exception as e:
            print(f"Error during search (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                sleep_time = 5 + random.random() * 3
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Could not complete search.")
                return []

if __name__ == "__main__":
    # Print LanceDB version
    print(f"LanceDB version: {lancedb.__version__}")
    
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embeddings.py <extracted_data_dir> [test_query]")
        sys.exit(1)
    
    extracted_dir = sys.argv[1]
    table = process_extractions_to_embeddings(extracted_dir)
    
    # If a test query is provided, test retrieval
    if len(sys.argv) > 2:
        test_query = ' '.join(sys.argv[2:])
        print(f"\nTesting retrieval with query: '{test_query}'")
        results = test_retrieval(test_query, table)