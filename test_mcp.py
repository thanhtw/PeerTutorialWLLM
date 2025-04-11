import requests

def test_mcp_server():
    """Test if MCP server is running and responding."""
    try:
        response = requests.get("http://localhost:8000/api/status", timeout=5)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error connecting to MCP server: {str(e)}")
        return False

if __name__ == "__main__":
    result = test_mcp_server()
    print(f"MCP server running correctly: {result}")