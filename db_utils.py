# db_utils.py
import lancedb
import os
import pandas as pd
from functools import lru_cache
from typing import List, Dict, Any
import numpy as np

@lru_cache(maxsize=1)
def get_db_connection(db_path="stockreports_db"):
    """Cached database connection to avoid repeated connections"""
    return lancedb.connect(db_path)

def batch_insert(table, data: List[Dict[str, Any]], batch_size=1000):
    """Insert data in batches for better performance"""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        table.add(pd.DataFrame(batch))

def get_db_stats(db_path="stockreports_db"):
    """Get statistics on the LanceDB database with optimized queries"""
    db = get_db_connection(db_path)
    
    if "stock_chunks" not in db.table_names():
        return {
            "total_chunks": 0,
            "unique_sources": 0,
            "sources": []
        }
    
    table = db.open_table("stock_chunks")
    
    # Use optimized query for statistics
    df = table.to_pandas(columns=["source_file"])
    source_counts = df.groupby("source_file").size()
    
    stats = {
        "total_chunks": len(df),
        "unique_sources": len(source_counts),
        "sources": [{
            "name": source,
            "chunks": count
        } for source, count in source_counts.items()]
    }
    
    return stats

def clear_database(db_path="stockreports_db", confirm=True):
    """Clear the database with proper resource cleanup"""
    if confirm:
        confirmation = input("Are you sure you want to clear the database? (y/n): ")
        if confirmation.lower() != 'y':
            return False
    
    try:
        db = get_db_connection(db_path)
        if "stock_chunks" in db.table_names():
            db.drop_table("stock_chunks")
            # Clear the connection cache
            get_db_connection.cache_clear()
            return True
        return False
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False

def optimize_table(db_path="stockreports_db"):
    """Optimize the database table for better query performance"""
    try:
        db = get_db_connection(db_path)
        if "stock_chunks" in db.table_names():
            table = db.open_table("stock_chunks")
            table.optimize()
            return True
        return False
    except Exception as e:
        print(f"Error optimizing table: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python db_utils.py [stats|clear]")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    if action == "stats":
        get_db_stats()
    elif action == "clear":
        clear_database()
    else:
        print(f"Unknown action: {action}")
        print("Available actions: stats, clear")