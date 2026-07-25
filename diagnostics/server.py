"""Minimal static file + JSON API server for viewer.html / token_viewer.html.

No third-party deps (stdlib only).

Usage:
    python server.py [--port 8888]

Endpoints:
    GET /api/files          -> list of relative .json file paths under this dir
    GET /api/load?path=...  -> raw JSON content of that file
    GET /<any>              -> static file serving (viewer.html, token_viewer.html, ...)
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, unquote, urlparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {".venv", "node_modules", ".git", "__pycache__"}


def list_json_files() -> list[str]:
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".json"):
                rel = os.path.relpath(os.path.join(dirpath, fn), ROOT_DIR)
                files.append(rel)
    return sorted(files)


def safe_path(rel_path: str) -> str | None:
    full = os.path.normpath(os.path.join(ROOT_DIR, rel_path))
    if not full.startswith(ROOT_DIR):
        return None
    return full


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/files":
            self._send_json(list_json_files())
            return

        if parsed.path == "/api/load":
            rel_path = unquote(parse_qs(parsed.query).get("path", [""])[0])
            full = safe_path(rel_path)
            if not full or not os.path.isfile(full):
                self._send_json({"error": "not found"}, status=404)
                return
            with open(full) as f:
                self._send_json(json.load(f))
            return

        # static file serving
        rel_path = parsed.path.lstrip("/") or "viewer.html"
        full = safe_path(rel_path)
        if not full or not os.path.isfile(full):
            self.send_response(404)
            self.end_headers()
            return
        content_type = "text/html" if full.endswith(".html") else "application/octet-stream"
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass  # quiet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving {ROOT_DIR}")
    print(f"  -> http://localhost:{args.port}/token_viewer.html")
    print(f"  -> http://localhost:{args.port}/viewer.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
