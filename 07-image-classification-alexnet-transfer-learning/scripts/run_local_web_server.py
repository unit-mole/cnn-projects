from __future__ import annotations
import argparse, http.server, socketserver
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/"web"
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--port",type=int,default=8000); args=p.parse_args();
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw): super().__init__(*a,directory=str(ROOT),**kw)
    with socketserver.TCPServer(("127.0.0.1",args.port),Handler) as server:
        print(f"Serving {ROOT} at http://127.0.0.1:{args.port}"); server.serve_forever()
