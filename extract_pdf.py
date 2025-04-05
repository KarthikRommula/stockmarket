# extract_pdf.py
import os
import io
import json
import time
import fitz
import boto3
import pickle
import hashlib
import numpy as np
import concurrent.futures
import multiprocessing as mp
from PIL import Image
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Generator, Optional
from pathlib import Path
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
from memory_profiler import profile
from threading import Lock

# Import embeddings module when needed
try:
    from embeddings import process_extractions_to_embeddings
except ImportError:
    # This allows extract_pdf.py to run independently without embeddings module
    print("Warning: embeddings module not available. Some functionality may be limited.")

# Load environment variables
load_dotenv()

# Configure AWS Bedrock client for Claude model access
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

# Get inference profile ID from environment variables
INFERENCE_PROFILE_ID = os.getenv('INFERENCE_PROFILE_ARN')

# Global settings for optimization
PDF_CHUNK_SIZE = 100  # Increased number of pages to process in each batch
MAX_WORKERS = min(64, (os.cpu_count() or 1) * 8)  # Increased worker count for better parallelization
OCR_TIMEOUT = 10  # Reduced timeout for OCR
EXTRACTION_TIMEOUT = 5  # Reduced timeout for text extraction
MEMORY_LIMIT = 0.9  # Increased memory usage limit (90% of available RAM)
BATCH_EMBEDDING_SIZE = 5  # Number of chunks to embed in a single batch
DISABLE_THROTTLING = True  # Set to True to disable throttling pauses

# Thread-safe progress tracking
progress_lock = Lock()
processed_pages = 0
start_time = time.time()

# Common Tesseract installation paths
TESSERACT_PATHS = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    r'C:\Users\KARTHIK\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
]

def find_tesseract():
    """Find Tesseract executable in common installation paths"""
    for path in TESSERACT_PATHS:
        if os.path.exists(path):
            return path
    return None

try:
    import pytesseract
    tesseract_path = find_tesseract()
    if tesseract_path:
        print(f"Found Tesseract at: {tesseract_path}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        TESSERACT_AVAILABLE = True
    else:
        print("Warning: Tesseract not found in common locations. Please install Tesseract-OCR.")
        print("Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        TESSERACT_AVAILABLE = False
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. OCR functionality will be disabled.")
except Exception as e:
    TESSERACT_AVAILABLE = False
    print(f"Warning: Error configuring Tesseract OCR: {e}")

def extract_text_from_image(image_data: bytes) -> Optional[str]:
    """Extract text from image using OCR with enhanced preprocessing"""
    if not TESSERACT_AVAILABLE:
        print("OCR skipped: Tesseract not available")
        return None
        
    try:
        print("Starting OCR process...")
        # Convert image bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        print(f"Image opened successfully. Mode: {image.mode}, Size: {image.size}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"Converted image to RGB mode")
        
        # Apply image enhancements
        from PIL import ImageEnhance
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Binarize image
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
        print("Applied image preprocessing")
        
        # Extract text using Tesseract with optimized settings
        print("Starting Tesseract OCR...")
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%&*()[]{}<>_+-=\'\"\n "'
        text = pytesseract.image_to_string(
            image,
            config=custom_config,
            lang='eng'
        )
        
        # Clean and validate the extracted text
        if text:
            text = text.strip()
            # Remove any non-printable characters
            text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
            # Remove excessive whitespace
            text = ' '.join(text.split())
            return text if len(text) > 10 else None  # Only return if we got meaningful text
        
        return None
    except Exception as e:
        print(f"OCR error: {e}")
        return None

@contextmanager
def safe_pdf_open(pdf_path: str) -> Generator[fitz.Document, None, None]:
    """Safely open and close PDF files with proper resource management"""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        yield doc
    finally:
        if doc:
            try:
                doc.close()
            except Exception:
                pass

# Initialize cache directory
CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / '.cache'
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(pdf_path: str) -> Path:
    """Generate cache file path for a PDF"""
    pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()
    return CACHE_DIR / f"{pdf_hash}.pickle"

def cache_exists(pdf_path: str) -> bool:
    """Check if cache exists for a PDF"""
    return get_cache_path(pdf_path).exists()

# Load environment variables
load_dotenv()

# Configure AWS Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION', 'ap-south-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

@lru_cache(maxsize=1000)
def extract_page_text(page, method="text", flags=fitz.TEXT_PRESERVE_WHITESPACE):
    """Extract text from a page using multiple methods with fallback"""
    try:
        text = ""
        errors = []
        
        # Try all methods in sequence until we get good text
        methods = [("text", flags), ("blocks", None), ("html", None)]
        for method_name, method_flags in methods:
            try:
                if method_name == "text":
                    text = page.get_text(flags=method_flags or flags)
                elif method_name == "blocks":
                    blocks = page.get_text("blocks")
                    if isinstance(blocks, list):
                        text = "\n".join(block[4] for block in blocks if block[4].strip())
                elif method_name == "html":
                    html = page.get_text("html")
                    if html:
                        import re
                        text = re.sub('<[^<]+?>', '', html)
                
                # Check if we got good text
                if text and len(text.strip()) > 20:  # Reasonable text found
                    return text
                
            except Exception as e:
                errors.append(f"{method_name} failed: {str(e)}")
                continue
        
        # If we got here, no method worked well
        if errors:
            print("Text extraction attempts failed:")
            for error in errors:
                print(f"  - {error}")
        
        # Return whatever text we got, even if it's not ideal
        return text
        
    except Exception as e:
        print(f"Text extraction error: {e}")
        return ""

@profile
def extract_pdf(pdf_path: str, batch_size: int = 50) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with page numbers using optimized batched processing.
    
    Args:
        pdf_path (str): Path to the PDF file
        batch_size (int): Number of pages to process in each batch
        
    Returns:
        list: List of dictionaries containing text and page number
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    extracted_pages = []
    start_time = time.time()
    
    try:
        with safe_pdf_open(pdf_path) as doc:
            if not doc:
                raise ValueError(f"Could not open PDF: {pdf_path}")
            
            if doc.needs_pass:
                raise ValueError(f"PDF is encrypted: {pdf_path}")
                
            total_pages = len(doc)
            if total_pages == 0:
                raise ValueError(f"PDF is empty: {pdf_path}")
            
            print(f"Processing {total_pages} pages...")
            
            # Use ThreadPoolExecutor since ProcessPool doesn't work well with PyMuPDF
            max_workers = min(32, (os.cpu_count() or 1) * 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Process pages in batches
                for start_idx in range(0, total_pages, batch_size):
                    end_idx = min(start_idx + batch_size, total_pages)
                    pages = [doc.load_page(i) for i in range(start_idx, end_idx)]
                    
                    # Submit page processing tasks
                    futures = []
                    for idx, page in enumerate(pages):
                        page_idx = start_idx + idx
                        futures.append(executor.submit(extract_page_text, page))
                    
                    # Process results as they complete
                    for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            text = future.result(timeout=30)
                            page = pages[idx]
                            page_idx = start_idx + idx
                            extracted_text = ""
                            method = "unknown"
                            
                            # 1. Try native text extraction
                            if text and text.strip():
                                extracted_text = text.strip()
                                method = "native"
                            
                            # 2. Try getting text blocks if native fails
                            if not extracted_text:
                                try:
                                    blocks = extract_page_text(page, "blocks")
                                    if isinstance(blocks, list):
                                        block_text = "\n".join(block[4] for block in blocks if block[4].strip())
                                        if block_text.strip():
                                            extracted_text = block_text.strip()
                                            method = "blocks"
                                except Exception as e:
                                    print(f"Block extraction failed: {e}")
                            
                            # 3. Try HTML extraction if blocks fail
                            if not extracted_text:
                                try:
                                    html_text = extract_page_text(page, "html")
                                    if html_text:
                                        import re
                                        clean_text = re.sub('<[^<]+?>', '', html_text)
                                        if clean_text.strip():
                                            extracted_text = clean_text.strip()
                                            method = "html"
                                except Exception as e:
                                    print(f"HTML extraction failed: {e}")
                            
                            # 4. Try OCR as last resort
                            if not extracted_text:
                                try:
                                    images = page.get_images()
                                    if images:
                                        for scale in [4, 2, 1]:
                                            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                                            ocr_text = extract_text_from_image(pix.tobytes())
                                            if ocr_text and ocr_text.strip():
                                                extracted_text = ocr_text.strip()
                                                method = "ocr"
                                                break
                                except Exception as e:
                                    print(f"OCR failed: {e}")
                            
                            # Add extracted text to batch if found
                            if extracted_text:
                                extracted_pages.append({
                                    "text": extracted_text,
                                    "page_number": page_idx + 1,
                                    "extraction_method": method
                                })
                        except Exception as e:
                            print(f"Error processing page {start_idx + idx + 1}: {e}")
                    
                    # Update progress
                    pages_processed = len(extracted_pages)
                    elapsed = time.time() - start_time
                    print(f"\rProcessed {pages_processed}/{total_pages} pages "
                          f"({pages_processed/elapsed:.1f} p/s)", end="")
            
            print("\nExtraction complete.")
            
            if not extracted_pages:
                raise ValueError(f"No text could be extracted from {pdf_path}. The file might be scanned, encrypted, or corrupted.")
            
            # Sort by page number
            extracted_pages.sort(key=lambda x: x['page_number'])
            return extracted_pages
                
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        raise
    

@profile
def chunk_text(extracted_pages: List[Dict[str, Any]], chunk_size: int = 4000, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Break extracted text into chunks with proper page metadata using optimized processing.
    
    Args:
        extracted_pages (list): List of dictionaries with text and page_number
        chunk_size (int): Target size for each chunk (default: 2000)
        chunk_overlap (int): Overlap between chunks (default: 100)
        
    Returns:
        list: List of chunk dictionaries with text and metadata
    """
    chunks = []
    sentence_endings = {'.', '!', '?'}
    
    def find_break_point(text: str, end: int, search_start: int) -> int:
        # Quick check for paragraph break first
        para_break = text.rfind('\n\n', search_start, end)
        if para_break != -1:
            return para_break + 2
            
        # Efficient sentence boundary detection
        for i in range(end - 1, search_start - 1, -1):
            if i < len(text) and text[i] in sentence_endings:
                # Check if followed by space or newline
                if i + 1 < len(text) and (text[i + 1].isspace() or text[i + 1] == '\n'):
                    return i + 2
        return end
    
    # Process pages in parallel
    def process_page(page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        page_chunks = []
        text = page_data.get("text", "").strip()
        page_num = page_data.get("page_number", 0)
        
        if not text:
            return []
            
        if len(text) <= chunk_size:
            return [{
                "text": text,
                "metadata": {
                    "pages": [page_num],
                    "source_page": page_num
                }
            }]
        
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            search_start = max(start, end - int(chunk_size * 0.2))
            
            # Find optimal break point
            break_point = find_break_point(text, end, search_start)
            
            chunk_text = text[start:break_point].strip()
            if chunk_text:
                page_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "pages": [page_num],
                        "source_page": page_num
                    }
                })
            
            # Move start position with overlap
            start = max(start + chunk_size - chunk_overlap, break_point)
        
        return page_chunks
    
    # Process pages in parallel using ThreadPoolExecutor - increased workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_page, page) for page in extracted_pages]
        
        for future in tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures),
                          desc="Chunking pages"):
            chunks.extend(future.result())
    
    print(f"Created {len(chunks)} chunks from {len(extracted_pages)} pages")
    
    # Debug output
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i} pages: {chunk['metadata']['pages']}")
    
    return chunks

@sleep_and_retry
@limits(calls=10, period=60)  # Increased rate limit: 10 calls per minute
def summarize_with_claude(chunk_text: str, max_retries: int = 3) -> str:
    """Summarize text using Claude with retry mechanism"""
    prompt = f"""Below is a section from a stock report. Please provide a concise summary that captures the key financial information, metrics, and insights:

{chunk_text}

Summary:"""

    for attempt in range(max_retries):
        try:
            # Use Claude 3.5 Sonnet with profile ID as model ID
            if INFERENCE_PROFILE_ID:
                print(f"Using Claude 3.5 Sonnet model: {INFERENCE_PROFILE_ID}")
                response = bedrock_runtime.invoke_model(
                    modelId=INFERENCE_PROFILE_ID,
                    body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
                response_body = json.loads(response['body'].read())
                summary = response_body['content'][0]['text']
                
                return summary
            else:
                # Fallback to Claude 3 Sonnet if no inference profile is set
                print("No inference profile found. Falling back to Claude 3 Sonnet.")
                print("Using Claude 3 Sonnet model: anthropic.claude-3-sonnet-20240229-v1:0")
                response = bedrock_runtime.invoke_model(
                    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 500,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                )
                
                response_body = json.loads(response['body'].read())
                summary = response_body['content'][0]['text']
                
                return summary
        
        except Exception as e:
            if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 2)  # Exponential backoff starting with 4 seconds
                print(f"Rate limited. Waiting {wait_time} seconds before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                raise

def save_partial_results(data, output_filename):
    """Save partial results to avoid losing progress"""
    with open(output_filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved partial results to {output_filename}")

@profile
def process_stock_report(pdf_path: str, output_dir: str = "extracted_data", batch_size: int = 50, max_workers: int = None) -> str:
    """Process a stock report PDF and save extracted data with caching"""
    # Validate input file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    # Check cache first
    cache_file = get_cache_path(pdf_path)
    if cache_exists(pdf_path):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if time.time() - cached_data.get('timestamp', 0) < 86400:  # 24 hour cache
                    print(f"Using cached results from {cache_file}")
                    return cached_data['output_filename']
                print("Cache expired, reprocessing...")
        except Exception as e:
            print(f"Cache read error: {e}. Processing PDF...")
    
    # Set optimal worker count - increased for better parallelization
    if max_workers is None:
        max_workers = min(64, (os.cpu_count() or 1) * 8)
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract text with layout information
    print(f"Extracting text from {pdf_path}...")
    extracted_pages = extract_pdf(pdf_path, batch_size=batch_size)
    
    # Chunk the extracted text
    print("Chunking text...")
    text_chunks = chunk_text(extracted_pages)
    
    # Prepare output file name
    output_filename = os.path.join(output_dir, os.path.basename(pdf_path).replace('.pdf', '.json'))
    result_data = {
        "source": pdf_path,
        "chunks": text_chunks
    }
    
    # Process chunks in parallel with progress tracking - optimized version
    print("Summarizing chunks with Claude...")
    successful_chunks = 0
    failed_chunks = []
    
    def process_chunk(chunk_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        try:
            if not chunk_data.get("text"):
                return False, chunk_data
                
            text = chunk_data["text"].strip()
            if not text:
                return False, chunk_data
            
            # Optimize text length for faster processing
            if len(text) > 8000:  # If text is too long
                # Extract key sections only
                sections = text.split('\n\n')
                if len(sections) > 10:
                    # Take first 3 paragraphs, last 3 paragraphs, and 4 from the middle
                    middle_start = len(sections) // 2 - 2
                    selected_sections = sections[:3] + sections[middle_start:middle_start+4] + sections[-3:]
                    text = '\n\n'.join(selected_sections)
            
            summary = summarize_with_claude(text)
            if summary:
                chunk_data["summary"] = summary.strip()
                return True, chunk_data
            return False, chunk_data
        except Exception as e:
            chunk_data["summary"] = f"Error: {str(e)}"
            return False, chunk_data
    
    # Process chunks in larger batches for better throughput
    # Increased batch size from 10 to 20 for better parallelization
    chunk_batches = [text_chunks[i:i + 20] for i in range(0, len(text_chunks), 20)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, 16)) as executor:
        futures = []
        for batch_idx, batch in enumerate(chunk_batches):
            print(f"Processing batch {batch_idx+1}/{len(chunk_batches)}")
            for chunk in batch:
                futures.append(executor.submit(process_chunk, chunk))
            
            # Process one batch at a time to avoid overwhelming the API
            batch_futures = futures[-len(batch):]
            
            with tqdm(total=len(batch_futures), desc=f"Batch {batch_idx+1}", unit="chunk") as pbar:
                for i, future in enumerate(concurrent.futures.as_completed(batch_futures)):
                    try:
                        success, processed_chunk = future.result(timeout=20)  # Reduced timeout for faster failures
                        if success:
                            successful_chunks += 1
                        else:
                            failed_chunks.append(batch_idx * 20 + i + 1)
                        
                        # Update the chunk in text_chunks
                        chunk_index = batch_idx * 20 + i
                        if chunk_index < len(text_chunks):
                            text_chunks[chunk_index] = processed_chunk
                        
                        # Save partial results less frequently to reduce I/O overhead
                        if (successful_chunks % 10 == 0) or (successful_chunks + len(failed_chunks) == len(text_chunks)):
                            save_partial_results(result_data, output_filename)
                            
                    except concurrent.futures.TimeoutError:
                        print(f"Chunk {batch_idx * 20 + i + 1} processing timed out")
                        failed_chunks.append(batch_idx * 20 + i + 1)
                    except Exception as e:
                        print(f"Error processing chunk {batch_idx * 20 + i + 1}: {e}")
                        failed_chunks.append(batch_idx * 20 + i + 1)
                    finally:
                        pbar.update(1)
            
            # Add a small pause between batches if not disabled
            if not DISABLE_THROTTLING and batch_idx < len(chunk_batches) - 1:
                time.sleep(1)
    
    # Verify final output
    with open(output_filename, 'r') as f:
        data = json.load(f)
        print("\nVerifying output data structure:")
        print(f"Total chunks: {len(data.get('chunks', []))}")
        for i, chunk in enumerate(data.get('chunks', [])[:3]):  # Show first 3 chunks
            pages = chunk.get('metadata', {}).get('pages', [])
            print(f"Chunk {i} pages: {pages}")
    
    print(f"Completed processing. {successful_chunks} of {len(text_chunks)} chunks processed successfully.")
    
    # Cache results to disk
    cache_file = get_cache_path(pdf_path)
    cache_data = {
        'timestamp': time.time(),
        'output_filename': output_filename
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Processing complete! Results saved to {output_filename}")
    return output_filename


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    process_stock_report(pdf_path)
    