import os
import sys

# Ensure root directory is on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app


class VercelPathMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # Vercel rewrites forward requests to /api/index.
        # Restore the original requested path from Vercel edge headers.
        orig = (
            environ.get("HTTP_X_FORWARDED_URI")
            or environ.get("HTTP_X_MATCHED_PATH")
            or environ.get("HTTP_X_NOW_ROUTE_MATCHES")
        )
        if orig:
            environ["PATH_INFO"] = orig.split("?")[0]
        elif environ.get("PATH_INFO", "").startswith("/api/index"):
            sub = environ["PATH_INFO"][len("/api/index"):] or "/"
            environ["PATH_INFO"] = sub

        return self.wsgi_app(environ, start_response)


app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
