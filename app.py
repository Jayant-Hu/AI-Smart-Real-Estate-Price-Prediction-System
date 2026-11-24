import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

backend_dir = os.path.join(BASE_DIR, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    os.chdir(BASE_DIR)
except Exception:
    pass

from backend.main import app

# Vercel entrypoint alias
app = app

if __name__ == "__main__":
    app.run()
