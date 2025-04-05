# evaluate_rag.py
import json
import time
from rag_system import StockReportRAG

def evaluate_system(test_queries, output_file="evaluation_results.json"):
    """Evaluate the RAG system with a set of test queries"""
    rag = StockReportRAG()
    
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\nProcessing test query {i+1}/{len(test_queries)}: '{query}'")
        
        # Measure response time
        start_time = time.time()
        response = rag.process_query(query)
        end_time = time.time()
        
        # Record results
        result = {
            "query": query,
            "response": response,
            "response_time_seconds": end_time - start_time
        }
        
        results.append(result)
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {output_file}")
    
    # Calculate average response time
    avg_time = sum(r["response_time_seconds"] for r in results) / len(results)
    print(f"Average response time: {avg_time:.2f} seconds")

if __name__ == "__main__":
    # Sample test queries - these should be tailored to your stock reports
    test_queries = [
        "What were the revenue figures for the last quarter?",
        "How is the company performing relative to its competitors?",
        "What are the main risk factors mentioned in the report?",
        "What is the company's guidance for the next fiscal year?",
        "How have profit margins changed over the past year?"
    ]
    
    evaluate_system(test_queries)