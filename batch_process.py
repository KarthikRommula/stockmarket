# batch_process.py
import os
import glob
from extract_pdf import process_stock_report

def batch_process_pdfs(pdf_dir, output_dir="extracted_data"):
    """Process all PDFs in a directory"""
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        print(f"\nProcessing {os.path.basename(pdf_path)}...")
        process_stock_report(pdf_path, output_dir)
    
    print("\nBatch processing complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python batch_process.py <pdf_directory>")
        sys.exit(1)
    
    pdf_dir = sys.argv[1]
    batch_process_pdfs(pdf_dir)