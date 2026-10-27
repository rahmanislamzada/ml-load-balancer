"""Shared HTTP backend implementation (raw asyncio sockets, no web framework).

Endpoints:
  GET /                    -> artificial delay, then "OK from <name> ..."
  GET /health              -> JSON health status (no delay)
  GET /set_delay?delay=0.2 -> change the base delay at runtime
"""
import asyncio
import json
import random
from urllib.parse import urlparse, parse_qs


class Backend:
    def __init__(self, name: str, delay: float, jitter: float = 0.0):
        self.name = name
        self.delay = delay      # base artificial delay (seconds)
        self.jitter = jitter    # extra random delay in [0, jitter]
        self.port = 0
        self.served = 0         # how many normal requests were handled

    def current_delay(self) -> float:
        return self.delay + random.uniform(0, self.jitter)

    def build_response(self, status: str, body: str,
                       content_type: str = "text/plain") -> bytes:
        """Build a raw HTTP/1.1 response. Content-Length is the real body size
        in BYTES, so the client knows exactly where the message ends."""
        payload = body.encode()
        headers = (
            f"HTTP/1.1 {status}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"X-Backend: {self.name}:{self.port}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        return headers.encode() + payload

    async def handle_client(self, reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter):
        try:
            # 1. Request line, e.g. "GET /set_delay?delay=0.2 HTTP/1.1"
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            if not request_line:
                return

            # 2. Consume headers until the blank line; b"" = client closed
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if line in (b"\r\n", b"\n", b""):
                    break

            # 3. Parse path and query string separately
            parts = request_line.decode(errors="replace").split()
            target = parts[1] if len(parts) >= 2 else "/"
            url = urlparse(target)

            if url.path == "/health":
                body = json.dumps({"status": "healthy", "backend": self.name,
                                   "port": self.port})
                response = self.build_response("200 OK", body, "application/json")

            elif url.path == "/set_delay":
                try:
                    value = float(parse_qs(url.query)["delay"][0])
                    self.delay = max(0.0, min(10.0, value))
                    response = self.build_response(
                        "200 OK", f"Delay of {self.name} set to {self.delay}s\n")
                except (KeyError, ValueError, IndexError):
                    response = self.build_response(
                        "400 Bad Request", "Use /set_delay?delay=0.2\n")

            else:
                d = self.current_delay()
                await asyncio.sleep(d)   # simulate a slow/fast/unstable server
                self.served += 1
                response = self.build_response(
                    "200 OK",
                    f"OK from {self.name} backend on port {self.port} "
                    f"(delay {d:.3f}s)\n")

            writer.write(response)
            await writer.drain()

        except (asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self, port: int):
        self.port = port
        server = await asyncio.start_server(self.handle_client, "0.0.0.0", port)
        print(f"{self.name.capitalize()} backend listening on port {port}")
        async with server:
            await server.serve_forever()