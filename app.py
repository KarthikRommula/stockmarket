# app.py
import os
import tempfile
import streamlit as st
import pandas as pd
from extract_pdf import process_stock_report
from embeddings import process_extractions_to_embeddings
from rag_system import StockReportRAG

# Set page configuration
st.set_page_config(
    page_title="Stock Reports Analysiss",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if 'rag_system' not in st.session_state:
    try:
        st.session_state.rag_system = StockReportRAG()
    except ValueError:
        st.session_state.rag_system = None

if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

if 'available_sources' not in st.session_state:
    st.session_state.available_sources = []
    if st.session_state.rag_system:
        st.session_state.available_sources = st.session_state.rag_system.get_source_information()

# Set up the UI
st.title("Stock Report Analysis")
st.write("Upload stock reports, process them, and ask questions about the content and more.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤Upload & Process", "🔍Query & Analysis", "⚙️System Information"])

with tab1:
    st.header("Upload Stock Reports")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF stock reports", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s):")
        for file in uploaded_files:
            st.write(f"- {file.name}")
        
        # Process button
        if st.button("Process Reports"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create temp directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted_dir = os.path.join(temp_dir, "extracted")
                os.makedirs(extracted_dir, exist_ok=True)
                
                # Process each file
                for i, file in enumerate(uploaded_files):
                    # Save uploaded file to temp directory
                    status_text.text(f"Saving {file.name}...")
                    pdf_path = os.path.join(temp_dir, file.name)
                    with open(pdf_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Extract text
                    status_text.text(f"Extracting text from {file.name}...")
                    progress_bar.progress((i * 3) / (len(uploaded_files) * 3))
                    
                    output_file = process_stock_report(pdf_path, extracted_dir)
                    progress_bar.progress((i * 3 + 1) / (len(uploaded_files) * 3))
                    
                    # Update status
                    status_text.text(f"Generating embeddings for {file.name}...")
                    
                # Generate embeddings for all extracted files
                status_text.text("Generating and storing embeddings...")
                process_extractions_to_embeddings(extracted_dir)
                progress_bar.progress(1.0)
                
                # Reinitialize RAG system
                status_text.text("Initializing RAG system...")
                st.session_state.rag_system = StockReportRAG()
                st.session_state.available_sources = st.session_state.rag_system.get_source_information()
                st.session_state.processing_complete = True
                
                status_text.text("Processing complete!")
                st.success("All reports have been processed and are ready for querying!")

with tab2:
    st.header("Query Stock Reports")
    
    if not st.session_state.rag_system:
        st.warning("No data available. Please upload and process stock reports first, or check that the database exists.")
    else:
        # Display available sources
        if st.session_state.available_sources:
            st.subheader("Available Reports")
            source_df = pd.DataFrame([
                {"Report": s["source"], "Chunks": s["chunks"]}
                for s in st.session_state.available_sources
            ])
            st.dataframe(source_df)
        
        # Query input
        query = st.text_input("Ask a question about the stock reports:", "What were the key financial highlights?")
        
        use_sources = st.checkbox("Show source information", value=True)
        
        if st.button("Submit Query"):
            if query:
                with st.spinner("Processing query..."):
                    # Get response from RAG system
                    response = st.session_state.rag_system.query(query, max_sources=3)
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(response["answer"])
                    
                    # Display sources if enabled
                    if use_sources and response.get("sources"):
                        st.subheader("Sources")
                        for i, source in enumerate(response["sources"]):
                            with st.expander(f"Source {i+1}: {source['source']} (Page {source['page']})"):
                                st.write(source["text"])
                                st.caption(f"Relevance score: {source['score']:.2f}")
            else:
                st.warning("Please enter a question to query the stock reports.")

with tab3:
    st.header("System Information")
    
    # Display system settings and statistics
    st.subheader("Database Statistics")
    if st.session_state.rag_system:
        stats = st.session_state.rag_system.get_system_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", stats["total_documents"])
        with col2:
            st.metric("Total Chunks", stats["total_chunks"])
        with col3:
            st.metric("Embedding Dimensions", stats["embedding_dimensions"])
        
        st.subheader("System Configuration")
        config = st.session_state.rag_system.get_config()
        
        # Fix: Ensure all values in the config dataframe are strings to avoid Arrow conversion issues
        config_df = pd.DataFrame([
            {"Setting": "Chunk Size", "Value": str(config["chunk_size"])},
            {"Setting": "Chunk Overlap", "Value": str(config["chunk_overlap"])},
            {"Setting": "Similarity Metric", "Value": str(config["similarity_metric"])},
            {"Setting": "Model", "Value": str(config["model"])}
        ])
        st.table(config_df)
    else:
        st.warning("RAG system not initialized. Please process documents first.")
    
    # Add reset functionality
    st.subheader("Reset System")
    if st.button("🗑️ Reset Database", type="primary", help="This will clear all processed documents from the database"):
        if st.session_state.rag_system:
            st.session_state.rag_system.reset_database()
            st.session_state.rag_system = None
            st.session_state.processing_complete = False
            st.session_state.available_sources = []
            st.success("Database has been reset. You can now upload new documents.")
            st.rerun()

# Add footer
# Add spacing before footer

# Create footer with horizontal line
# Add footer
# Create footer with horizontal line
st.markdown("---")

# Centered copyright caption at the very bottom
st.markdown("""
<div style="text-align: center; color: black; font-size: 0.8rem; margin-bottom: 0px; position: fixed; bottom: 0; left: 0; right: 0; background-color: white; padding: 5px;">
    © 2025 Stock Report Analysis Tool
</div>
""", unsafe_allow_html=True)
