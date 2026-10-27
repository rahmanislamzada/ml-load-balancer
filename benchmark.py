"""Benchmark: round-robin vs least-connections vs ML under 3 scenarios.

Scenarios:
  steady   - constant load
  burst    - low load -> sudden high load -> low load
  slowdown - the 'fast' backend becomes slow (0.1 s -> 0.8 s) in the middle

Prerequisites: backends running (Terminal 1). Stop any proxy running in
Terminal 2: this script starts and stops the proxy itself (port 8080).

Run:  python benchmark.py      (takes several minutes)
Output: results/raw.csv, results/summary.csv, results/*.png
"""
import asyncio
import os
import socket
import subprocess
import sys
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROXY_PORT = 8080
STRATEGIES = ["rr", "lc", "ml"]
REPS = 3                                   # repetitions per (scenario, strategy)
DEFAULT_DELAYS = {9001: 0.5, 9002: 0.1, 9003: 0.2}   # slow, fast, variable
RESULTS_DIR = "results"

# scenario -> list of phases: (phase name, requests, concurrency, action before phase)
SCENARIOS = {
    "steady":   [("all", 300, 20, None)],
    "burst":    [("low-a", 100, 5, None), ("high", 300, 40, None), ("low-b", 100, 5, None)],
    "slowdown": [("before", 200, 20, None), ("after", 400, 20, (9002, 0.8))],
}


# ------------------------------------------------------------------ helpers
def set_delay(port: int, delay: float) -> None:
    urllib.request.urlopen(
        f"http://127.0.0.1:{port}/set_delay?delay={delay}", timeout=5).read()


def reset_delays() -> None:
    for port, delay in DEFAULT_DELAYS.items():
        set_delay(port, delay)


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def start_proxy(strategy: str) -> subprocess.Popen:
    if port_open(PROXY_PORT):
        sys.exit("Port 8080 is busy. Stop the proxy in Terminal 2 (Ctrl+C) and retry.")
    proc = subprocess.Popen(
        [sys.executable, "-m", "load_balancer.proxy", "--strategy", strategy],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):                     # wait up to ~15 s until it listens
        if port_open(PROXY_PORT):
            return proc
        time.sleep(0.25)
    proc.terminate()
    sys.exit(f"Proxy ({strategy}) did not start.")


def stop_proxy(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(0.5)


# ------------------------------------------------------------- load client
async def one_request(t0: float):
    start = time.perf_counter()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", PROXY_PORT), 5)
        writer.write(b"GET / HTTP/1.1\r\nHost: bench\r\nConnection: close\r\n\r\n")
        await writer.drain()
        data = await asyncio.wait_for(reader.read(-1), 20)
        writer.close()
        latency = time.perf_counter() - start
        backend = "unknown"
        for line in data.decode(errors="replace").split("\r\n"):
            if line.lower().startswith("x-backend:"):
                backend = line.split(":", 1)[1].strip().split(":")[0]
        return start - t0, backend, latency, data.startswith(b"HTTP/1.1 200")
    except Exception:
        return start - t0, "error", time.perf_counter() - start, False


async def run_phase(n: int, concurrency: int, t0: float):
    sem = asyncio.Semaphore(concurrency)

    async def limited():
        async with sem:
            return await one_request(t0)

    return await asyncio.gather(*(limited() for _ in range(n)))


async def run_scenario(strategy: str, scenario: str, rep: int):
    rows = []
    t0 = time.perf_counter()
    for phase, n, concurrency, action in SCENARIOS[scenario]:
        if action:                                  # e.g. slow down the fast backend
            await asyncio.to_thread(set_delay, *action)
        p_start = time.perf_counter()
        results = await run_phase(n, concurrency, t0)
        elapsed = time.perf_counter() - p_start
        for t, backend, latency, ok in results:
            rows.append(dict(strategy=strategy, scenario=scenario, rep=rep, phase=phase,
                             t=round(t, 4), backend=backend, latency=round(latency, 4),
                             ok=int(ok), phase_time=round(elapsed, 4)))
    return rows


# -------------------------------------------------------------------- stats
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    per_run = []
    for (scenario, phase, strategy, rep), g in df.groupby(
            ["scenario", "phase", "strategy", "rep"], sort=False):
        ok = g[g.ok == 1]
        lats = ok.latency.to_numpy()
        per_run.append(dict(
            scenario=scenario, phase=phase, strategy=strategy,
            avg_ms=lats.mean() * 1000 if len(lats) else np.nan,
            p95_ms=np.percentile(lats, 95) * 1000 if len(lats) else np.nan,
            p99_ms=np.percentile(lats, 99) * 1000 if len(lats) else np.nan,
            throughput=len(g) / g.phase_time.iloc[0],
            error_pct=100 * (len(g) - len(ok)) / len(g),
            fast_share_pct=100 * (ok.backend == "fast").sum() / max(len(ok), 1)))
    metrics = ["avg_ms", "p95_ms", "p99_ms", "throughput", "error_pct", "fast_share_pct"]
    return (pd.DataFrame(per_run)
            .groupby(["scenario", "phase", "strategy"], sort=False)[metrics]
            .mean().reset_index())


# ------------------------------------------------------------------- charts
def make_charts(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    keys = [(s, p[0]) for s, phases in SCENARIOS.items() for p in phases]
    labels = [f"{s}\n{p}" for s, p in keys]
    x = np.arange(len(keys))
    width = 0.27

    # 1) avg / p95 / throughput per scenario phase
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    for ax, (metric, title) in zip(axes, [("avg_ms", "Average latency (ms) - lower is better"),
                                          ("p95_ms", "p95 latency (ms) - lower is better"),
                                          ("throughput", "Throughput (req/s) - higher is better")]):
        for i, strat in enumerate(STRATEGIES):
            vals = [summary[(summary.scenario == s) & (summary.phase == p) &
                            (summary.strategy == strat)][metric].iloc[0] for s, p in keys]
            ax.bar(x + (i - 1) * width, vals, width, label=strat)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "metrics_by_phase.png"), dpi=130)
    plt.close(fig)

    # 2) latency over time in the slowdown scenario (rolling mean, repetition 0)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"rr": "tab:blue", "lc": "tab:orange", "ml": "tab:green"}
    for strat in STRATEGIES:
        g = df[(df.scenario == "slowdown") & (df.strategy == strat) &
               (df.rep == 0) & (df.ok == 1)].sort_values("t")
        ax.plot(g.t, g.latency.rolling(20, min_periods=1).mean() * 1000,
                label=strat, color=colors[strat])
        after_start = df[(df.scenario == "slowdown") & (df.strategy == strat) &
                         (df.rep == 0) & (df.phase == "after")].t.min()
        ax.axvline(after_start, color=colors[strat], linestyle=":", alpha=0.6)
    ax.set_title("Slowdown scenario: 'fast' backend becomes slow (dotted line)")
    ax.set_xlabel("time since scenario start (s)")
    ax.set_ylabel("latency, rolling mean of 20 requests (ms)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "slowdown_over_time.png"), dpi=130)
    plt.close(fig)

    # 3) backend distribution after the slowdown
    after = df[(df.scenario == "slowdown") & (df.phase == "after") & (df.ok == 1)]
    share = (after.groupby(["strategy", "backend"]).size()
             .unstack(fill_value=0).reindex(STRATEGIES))
    share = share.div(share.sum(axis=1), axis=0) * 100
    ax = share.plot(kind="bar", stacked=True, figsize=(7, 5), rot=0)
    ax.set_title("Backend share (%) after 'fast' slowed down")
    ax.set_ylabel("% of requests")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "distribution_after_slowdown.png"), dpi=130)
    plt.close()


# --------------------------------------------------------------------- main
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        reset_delays()
    except Exception:
        sys.exit("Backends are not reachable. Run 'python start_backends.py' in Terminal 1.")

    all_rows = []
    total = len(SCENARIOS) * REPS * len(STRATEGIES)
    done = 0
    for scenario in SCENARIOS:
        for rep in range(REPS):
            for strategy in STRATEGIES:        # interleaved to reduce time drift
                done += 1
                print(f"[{done}/{total}] scenario={scenario} rep={rep + 1} strategy={strategy}")
                reset_delays()
                proc = start_proxy(strategy)
                try:
                    all_rows += asyncio.run(run_scenario(strategy, scenario, rep))
                finally:
                    stop_proxy(proc)
    reset_delays()

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "raw.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print("\n=== SUMMARY (mean of", REPS, "repetitions) ===")
    print(summary.round(1).to_string(index=False))
    make_charts(df, summary)
    print(f"\nCharts and CSV files saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()