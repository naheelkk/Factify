import subprocess
import webbrowser
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
backend_app = "backend.main:app"
frontend_app = os.path.join(BASE_DIR, "frontend", "app.py")

# Start backend (FastAPI on port 8000)
backend = subprocess.Popen([
    "uvicorn", backend_app,
    "--host", "127.0.0.1", "--port", "8000"
])

# Start frontend (Streamlit on port 8501)
frontend = subprocess.Popen([
    "streamlit", "run", frontend_app,
    "--server.port", "8501"
])

# Give Streamlit a moment to boot
time.sleep(5)
webbrowser.open("http://localhost:8501")

# Wait until one process exits
backend.wait()
frontend.wait()
