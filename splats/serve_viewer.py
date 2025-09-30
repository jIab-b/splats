import http.server
import socketserver
import argparse
from pathlib import Path
import os

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()
    os.chdir(args.dir)
    with socketserver.TCPServer(('', args.port), CORSRequestHandler) as httpd:
        print(f"Serving {args.dir} at http://localhost:{args.port}/viewer.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()


