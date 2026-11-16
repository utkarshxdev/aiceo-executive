import os
import sys
from urllib.parse import parse_qs

# Ensure root directory is on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app


class VercelPathMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        qs = parse_qs(environ.get("QUERY_STRING", ""))
        if "__path" in qs and qs["__path"][0]:
            val = qs["__path"][0]
            environ["PATH_INFO"] = "/" + val.lstrip("/")
        elif environ.get("HTTP_X_FORWARDED_URI"):
            environ["PATH_INFO"] = environ["HTTP_X_FORWARDED_URI"].split("?")[0]
        elif environ.get("PATH_INFO") in ("/api/index", "/api/index.py"):
            environ["PATH_INFO"] = "/"
        elif environ.get("PATH_INFO", "").startswith("/api/index"):
            environ["PATH_INFO"] = environ["PATH_INFO"][len("/api/index"):] or "/"

        return self.wsgi_app(environ, start_response)


app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
