"""Entry point: run the MCP server."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp_server.server import main

if __name__ == "__main__":
    main()
