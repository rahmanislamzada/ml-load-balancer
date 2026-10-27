"""Asyncio reverse proxy / load balancer on raw TCP sockets.

Flow for every client request:
  1. Accept a TCP connection from the client (L4)
  2. Read and parse the HTTP request manually (L7)
  3. Ask the strategy which backend to use
  4. Open a NEW TCP connection to that backend, forward the request
  5. Relay the response back to the client
  6. On failure (timeout / refused / empty reply) retry on another backend

Run from the project root:  python -m load_balancer.proxy --strategy rr
"""
import argparse
import asyncio
import csv
import inspect
import os
import time
from typing import Dict, Optional

from load_balancer.strategies import (
    BackendInfo, RoundRobinStrategy, LeastConnectionsStrategy, MLStrategy,
    RandomStrategy, FEATURE_NAMES)

PROXY_HOST = "0.0.0.0"
PROXY_PORT = 8080
HEADER_TIMEOUT = 5.0     # max time to receive the client's request headers
CONNECT_TIMEOUT = 3.0    # max time to open a TCP connection to a backend
READ_TIMEOUT = 10.0      # max time to wait for the backend's full response
HEALTH_INTERVAL = 2.0    # seconds between health checks

# One persistent object per backend: strategies read its live state.
BACKENDS: Dict[str, BackendInfo] = {
    "slow": BackendInfo("127.0.0.1", 9001, "slow"),
    "fast": BackendInfo("127.0.0.1", 9002, "fast"),
    "variable": BackendInfo("127.0.0.1", 9003, "variable"),
}

STRATEGY_CLASSES = {
    "rr": RoundRobinStrategy,
    "lc": LeastConnectionsStrategy,
    "ml": MLStrategy,
    "random": RandomStrategy,
}

# ---- training-data logging (enabled with --log-data) ----
DATA_FILE = os.path.join("ml", "data.csv")
LOG_DATA = False
_request_times: Dict[str, list] = {}   # per-backend send timestamps (for request_rate)


def current_request_rate(name: str, now: float) -> float:
    q = _request_times.setdefault(name, [])
    while q and now - q[0] > 10.0:
        q.pop(0)
    return len(q) / 10.0


def log_sample(row: list) -> None:
    new_file = not os.path.exists(DATA_FILE)
    with open(DATA_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["backend"] + FEATURE_NAMES + ["latency"])
        w.writerow(row)


# ---------------------------------------------------------------- HTTP parsing
async def read_request(reader: asyncio.StreamReader, client_ip: str) -> Optional[tuple]:
    """Read one HTTP request. Returns (request_line, raw_bytes_to_forward)."""
    try:
        # Headers end with an empty line: \r\n\r\n
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEADER_TIMEOUT)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
        return None

    lines = head.decode("latin-1").split("\r\n")
    request_line = lines[0]
    if len(request_line.split()) != 3:
        return None  # not "METHOD PATH HTTP/1.x"

    headers = []
    content_length = 0
    for line in lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(":")
        lname = name.strip().lower()
        if lname in ("connection", "proxy-connection", "keep-alive"):
            continue  # we always force Connection: close towards the backend
        if lname == "content-length":
            try:
                content_length = int(value.strip())
            except ValueError:
                return None
        headers.append(line)

    # Request body (e.g. POST) is separate from the headers
    body = b""
    if content_length > 0:
        try:
            body = await asyncio.wait_for(reader.readexactly(content_length), HEADER_TIMEOUT)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            return None

    headers.append(f"X-Forwarded-For: {client_ip}")
    headers.append("Connection: close")
    raw = (request_line + "\r\n" + "\r\n".join(headers) + "\r\n\r\n").encode("latin-1") + body
    return request_line, raw


# -------------------------------------------------------------- backend access
async def forward(backend: BackendInfo, request: bytes) -> bytes:
    """Open a new TCP connection to the backend, send the request, read the reply."""
    b_reader, b_writer = await asyncio.wait_for(
        asyncio.open_connection(backend.host, backend.port), CONNECT_TIMEOUT)
    try:
        b_writer.write(request)
        await b_writer.drain()
        # read(-1) = read until the backend closes the connection (EOF)
        response = await asyncio.wait_for(b_reader.read(-1), READ_TIMEOUT)
        if not response:
            raise ConnectionError("backend closed without sending a response")
        return response
    finally:
        b_writer.close()
        try:
            await b_writer.wait_closed()
        except Exception:
            pass


def record_result(backend: BackendInfo, latency: float, ok: bool) -> None:
    """Update the live statistics the strategies use for decisions."""
    if ok:
        backend.recent_latencies.append(latency)
        backend.recent_latencies = backend.recent_latencies[-20:]
        backend.avg_latency = sum(backend.recent_latencies) / len(backend.recent_latencies)
    # exponential moving average of the error rate
    backend.error_rate = 0.9 * backend.error_rate + 0.1 * (0.0 if ok else 1.0)


async def choose_backend(strategy, candidates: Dict[str, BackendInfo]) -> Optional[BackendInfo]:
    """Ask the strategy; fall back to least-connections if it fails."""
    try:
        result = strategy.select_backend(candidates)
        if inspect.isawaitable(result):
            result = await result
        if result is not None and result.name in candidates:
            return candidates[result.name]
    except Exception as e:
        print(f"[proxy] strategy error: {e!r}, falling back to least-connections")
    return min(candidates.values(), key=lambda b: b.active_connections, default=None)


# --------------------------------------------------------------- client handler
async def handle_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter, strategy) -> None:
    peer = writer.get_extra_info("peername")
    client_ip = peer[0] if peer else "unknown"
    try:
        parsed = await read_request(reader, client_ip)
        if parsed is None:
            return
        request_line, raw_request = parsed

        tried = set()
        while True:
            candidates = {n: b for n, b in BACKENDS.items()
                          if b.is_healthy and n not in tried}
            if not candidates:
                break
            backend = await choose_backend(strategy, candidates)
            if backend is None:
                break
            tried.add(backend.name)

                        # snapshot of the state AT DECISION TIME (these are the ML features)
            now = time.time()
            features = [float(backend.active_connections), backend.avg_latency,
                        backend.error_rate, current_request_rate(backend.name, now)]
            _request_times.setdefault(backend.name, []).append(now)

            backend.active_connections += 1
            start = time.perf_counter()
            try:
                response = await forward(backend, raw_request)
            except (asyncio.TimeoutError, OSError, ConnectionError) as e:
                record_result(backend, 0.0, ok=False)
                print(f"[proxy] {backend.name}:{backend.port} failed ({e!r}), retrying")
                continue  # retry on another backend
            finally:
                backend.active_connections -= 1

            latency = time.perf_counter() - start
            if LOG_DATA:
                log_sample([backend.name] + features + [round(latency, 4)])
            record_result(backend, latency, ok=True)
            try:
                strategy.update_backend_state(backend.name, backend)
            except Exception:
                pass
            writer.write(response)
            await writer.drain()
            print(f"[proxy] {request_line} -> {backend.name}:{backend.port} "
                  f"{latency * 1000:.0f} ms")
            return

        # all backends failed or are down
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n"
                     b"Connection: close\r\n\r\n")
        await writer.drain()
        print(f"[proxy] {request_line} -> 502 (no backend available)")
    except (ConnectionError, asyncio.CancelledError):
        pass  # client disconnected
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# ----------------------------------------------------------------- health check
async def check_backend(backend: BackendInfo) -> bool:
    try:
        r, w = await asyncio.wait_for(
            asyncio.open_connection(backend.host, backend.port), 1.0)
        w.write(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        await w.drain()
        data = await asyncio.wait_for(r.read(-1), 2.0)
        w.close()
        return data.startswith(b"HTTP/1.1 200")
    except Exception:
        return False


async def health_check_loop() -> None:
    """Periodically mark backends up/down (also brings recovered ones back)."""
    while True:
        results = await asyncio.gather(*(check_backend(b) for b in BACKENDS.values()))
        for backend, ok in zip(BACKENDS.values(), results):
            if ok != backend.is_healthy:
                print(f"[health] {backend.name}:{backend.port} is now "
                      f"{'UP' if ok else 'DOWN'}")
            backend.is_healthy = ok
        await asyncio.sleep(HEALTH_INTERVAL)


# ------------------------------------------------------------------------ main
async def main() -> None:
    parser = argparse.ArgumentParser(description="Load Balancer Proxy")
    parser.add_argument("--strategy", choices=list(STRATEGY_CLASSES), default="rr")
    parser.add_argument("--log-data", action="store_true",
                    help="append (features, latency) samples to ml/data.csv")
    args = parser.parse_args()
    global LOG_DATA
    LOG_DATA = args.log_data
    strategy = STRATEGY_CLASSES[args.strategy]()
    LOG_DATA = args.log_data

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, strategy), PROXY_HOST, PROXY_PORT)
    asyncio.create_task(health_check_loop())
    print(f"Proxy listening on port {PROXY_PORT} with strategy '{args.strategy}'")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())