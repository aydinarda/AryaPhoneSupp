"""Run the Arya Phone API server. Execute with: python run_server.py"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
