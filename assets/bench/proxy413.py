"""Forward /v1/systemone to a model server, turning its context-length rejection into HTTP 413.

pplx-decider's server answers a prompt over its 8,192-token limit with HTTP 400 and the error type
`exceed_context_size_error`. ruling's harness marks a record unscored on 413 and aborts the run on 400,
so this proxy maps that one rejection to 413. Every other request and response passes through unchanged.

    python3 assets/bench/proxy413.py <listen port> <upstream port>
"""
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN, UPSTREAM = int(sys.argv[1]), int(sys.argv[2])


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        request = urllib.request.Request(f"http://127.0.0.1:{UPSTREAM}{self.path}", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                status, payload = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, payload = error.code, error.read()
            if status == 400 and b"exceed_context_size_error" in payload:
                status = 413
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{UPSTREAM}{self.path}", timeout=60) as response:
            payload = response.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        if "413" in str(args):
            sys.stderr.write("413 (too long for the model): %s\n" % json.dumps(args))


ThreadingHTTPServer(("127.0.0.1", LISTEN), Handler).serve_forever()
