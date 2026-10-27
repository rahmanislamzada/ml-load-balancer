"""One-command demo: starts the 3 backends and the proxy in ONE process,
runs a load test for each strategy and prints a comparison table.

Examples (from the project root):
  python demo.py                              # steady + slowdown, rr/lc/ml
  python demo.py --scenario steady --requests 200
  python demo.py --serve ml                   # just run backends + ML proxy on :8080
"""
import argparse
import asyncio
import contextlib
import io
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # so ml/model.joblib is found

from backends import fast_backend, slow_backend, variable_backend
from load_balancer import proxy
from load_generator import one_request

DEFAULT_DELAYS = (0.5, 0.1, 0.2)   # slow, fast, variable(base)


def reset_state() -> None:
    """Default backend speeds + clean live statistics (fair start for each run)."""
    slow_backend.backend.delay, fast_backend.backend.delay, variable_backend.backend.delay = DEFAULT_DELAYS
    for b in proxy.BACKENDS.values():
        b.active_connections = 0
        b.avg_latency = 0.0
        b.error_rate = 0.0
        b.recent_latencies = []
        b.is_healthy = True


async def run_phase(n: int, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def limited():
        async with sem:
            return await one_request("127.0.0.1", proxy.PROXY_PORT)

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):   # hide per-request proxy logs
        results = await asyncio.gather(*(limited() for _ in range(n)))
    total = time.perf_counter() - t0

    lats = sorted(l for _, l, ok in results if ok)
    fast = sum(1 for b, _, ok in results if ok and b.startswith("fast"))
    return {
        "avg": sum(lats) / len(lats) * 1000 if lats else float("nan"),
        "p95": lats[max(int(len(lats) * 0.95) - 1, 0)] * 1000 if lats else float("nan"),
        "tput": n / total,
        "fast_pct": 100 * fast / max(len(lats), 1),
        "err": n - len(lats),
    }


async def with_proxy(key: str, coro_factory):
    """Start the proxy with one strategy, run something, stop the proxy."""
    strategy = proxy.STRATEGY_CLASSES[key]()
    server = await asyncio.start_server(
        lambda r, w: proxy.handle_client(r, w, strategy), "127.0.0.1", proxy.PROXY_PORT)
    try:
        return await coro_factory()
    finally:
        server.close()
        await server.wait_closed()


async def main(args) -> None:
    backend_tasks = [asyncio.create_task(slow_backend.start_server(9001)),
                     asyncio.create_task(fast_backend.start_server(9002)),
                     asyncio.create_task(variable_backend.start_server(9003))]
    await asyncio.sleep(0.5)
    health = asyncio.create_task(proxy.health_check_loop())

    if args.serve:
        strategy = proxy.STRATEGY_CLASSES[args.serve]()
        server = await asyncio.start_server(
            lambda r, w: proxy.handle_client(r, w, strategy), "0.0.0.0", proxy.PROXY_PORT)
        print(f"Proxy on http://127.0.0.1:{proxy.PROXY_PORT} with strategy '{args.serve}'. Ctrl+C to stop.")
        async with server:
            await server.serve_forever()
        return

    scenarios = ["steady", "slowdown"] if args.scenario == "both" else [args.scenario]
    rows = []
    for scenario in scenarios:
        for key in args.strategies:
            reset_state()
            print(f"running {scenario} / {key} ...")

            async def work():
                if scenario == "steady":
                    return [("all", await run_phase(args.requests, args.concurrency))]
                before = await run_phase(args.requests, args.concurrency)
                fast_backend.backend.delay = 0.8          # the 'fast' backend degrades
                after = await run_phase(args.requests * 2, args.concurrency)
                return [("before", before), ("after", after)]

            for phase, stats in await with_proxy(key, work):
                rows.append((scenario, phase, key, stats))
    reset_state()

    print("\nscenario  phase   strategy  avg(ms)  p95(ms)  req/s   fast%  errors")
    print("-" * 66)
    for scenario, phase, key, s in rows:
        print(f"{scenario:9s} {phase:7s} {key:9s} {s['avg']:7.0f}  {s['p95']:7.0f}  "
              f"{s['tput']:6.1f}  {s['fast_pct']:5.1f}  {s['err']:5d}")
    print("\nrr = round-robin, lc = least-connections, ml = ML-based. "
          "'slowdown/after': the fast backend went from 0.1 s to 0.8 s.")

    health.cancel()
    for t in backend_tasks:
        t.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ML load balancer demo")
    p.add_argument("--strategies", nargs="+", default=["rr", "lc", "ml"],
                   choices=list(proxy.STRATEGY_CLASSES))
    p.add_argument("--scenario", choices=["steady", "slowdown", "both"], default="both")
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--serve", choices=list(proxy.STRATEGY_CLASSES),
                   help="only run backends + proxy with this strategy (for curl/manual tests)")
    try:
        asyncio.run(main(p.parse_args()))
    except KeyboardInterrupt:
        print("\nStopped.")