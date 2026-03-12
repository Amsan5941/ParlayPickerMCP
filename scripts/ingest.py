"""Entry point: ingest CSV data into DuckDB."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ingest import ingest_all

if __name__ == "__main__":
    force = "--force" in sys.argv
    print("Starting data ingestion...")
    con = ingest_all(force=force)
    print("Done! Tables in database:")
    tables = con.execute("SHOW TABLES").fetchdf()
    print(tables.to_string(index=False))
    con.close()
