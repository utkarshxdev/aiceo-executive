import os
import sys
from urllib.parse import parse_qs

# Ensure root directory is on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app as flask_app


class VercelApp:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        path = "/"
        if "__path" in qs and qs["__path"][0]:
            path = "/" + qs["__path"][0].lstrip("/")
        elif environ.get("HTTP_X_FORWARDED_URI"):
            path = environ["HTTP_X_FORWARDED_URI"].split("?")[0]
        elif environ.get("PATH_INFO") in ("/api/index", "/api/index.py"):
            path = "/"
        elif environ.get("PATH_INFO", "").startswith("/api/index"):
            path = environ["PATH_INFO"][len("/api/index"):] or "/"

        environ["PATH_INFO"] = path
        if "REQUEST_URI" in environ:
            environ["REQUEST_URI"] = path
        if "RAW_URI" in environ:
            environ["RAW_URI"] = path

        print(f"[VERCEL WSGI] Routed to {path}", flush=True)
        return self.wsgi_app(environ, start_response)


# Export the wrapped WSGI application as `app` for Vercel
app = VercelApp(flask_app)
