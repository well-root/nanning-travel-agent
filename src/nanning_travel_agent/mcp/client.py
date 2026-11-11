#接入mcp
import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


#fu复制之前写的mcp的环境
METRO_PYTHON = os.environ.get(
    "NANNING_METRO_PYTHON",
    r"D:\agent\nanning-metro-mcp\.venv\Scripts\python.exe",
)
METRO_SERVER = os.environ.get(
    "NANNING_METRO_SERVER",
    r"D:\agent\nanning-metro-mcp\src\nanning_metro_mcp\server.py",
)

def _client()->MultiServerMCPClient:
    return MultiServerMCPClient({
        "nanning_metro": {
            "transport": "stdio",
            "command": METRO_PYTHON,
            "args": [METRO_SERVER],
            "env": {"PYTHONIOENCODING": "utf-8"},
    }
    }

    )

async def load_metro_tools()->list[BaseTool]:
    return await _client().get_tools()


