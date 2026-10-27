"""Simple async load generator: sends N requests with given concurrency,
then prints how many went to each backend and latency statistics."""
import argparse
import asyncio
import time
from collections import Counter


async def one_request(host: str, port: int):
    start = time.perf_counter()
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 5)
        writer.write(b"GET / HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n")
        await writer.drain()
        data = await asyncio.wait_for(reader.read(-1), 15)
        writer.close()
        latency = time.perf_counter() - start
        backend = "unknown"
        for line in data.decode(errors="replace").split("\r\n"):
            if line.lower().startswith("x-backend:"):
                backend = line.split(":", 1)[1].strip()
        ok = data.startswith(b"HTTP/1.1 200")
        return backend, latency, ok
    except Exception:
        return "error", time.perf_counter() - start, False


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    sem = asyncio.Semaphore(args.concurrency)

    async def limited():
        async with sem:
            return await one_request("127.0.0.1", args.port)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(limited() for _ in range(args.requests)))
    total = time.perf_counter() - t0

    counts = Counter(b for b, _, _ in results)
    lats = sorted(l for _, l, ok in results if ok)
    errors = sum(1 for _, _, ok in results if not ok)

    print(f"Requests: {args.requests}, concurrency: {args.concurrency}, total time: {total:.2f}s")
    print("Distribution:", dict(counts))
    if lats:
        print(f"avg latency: {sum(lats)/len(lats)*1000:.0f} ms | "
              f"p95: {lats[int(len(lats)*0.95)-1]*1000:.0f} ms | "
              f"throughput: {args.requests/total:.1f} req/s | errors: {errors}")


if __name__ == "__main__":
    asyncio.run(main())