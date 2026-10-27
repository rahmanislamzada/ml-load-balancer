"""Collect diverse training data: varies load and backend speeds.

Prerequisites: backends running (Terminal 1) and the proxy running with
    python -m load_balancer.proxy --strategy random --log-data
"""
import subprocess
import sys
import urllib.request

BACKEND_PORTS = {"slow": 9001, "fast": 9002, "variable": 9003}


def set_delay(port: int, delay: float) -> None:
    urllib.request.urlopen(f"http://127.0.0.1:{port}/set_delay?delay={delay}").read()


def run_load(requests: int, concurrency: int) -> None:
    subprocess.run([sys.executable, "load_generator.py",
                    "--requests", str(requests), "--concurrency", str(concurrency)])


# (slow, fast, variable-base) delays; variable keeps its random jitter on top
SCENARIOS = [
    (0.5, 0.1, 0.2),   # default
    (0.1, 0.5, 0.2),   # fast and slow swapped
    (0.3, 0.3, 0.1),   # all similar
    (0.8, 0.2, 0.4),   # everything heavier
    (0.1, 0.1, 0.1),   # everything quick
    (0.6, 0.05, 0.3),  # fast very fast
]
LOADS = [(150, 5), (200, 20), (200, 40)]  # (requests, concurrency)

if __name__ == "__main__":
    for slow, fast, var in SCENARIOS:
        set_delay(9001, slow)
        set_delay(9002, fast)
        set_delay(9003, var)
        for reqs, conc in LOADS:
            print(f"\n--- delays slow={slow} fast={fast} variable={var} | "
                  f"{reqs} requests, concurrency {conc} ---")
            run_load(reqs, conc)
    # restore defaults
    set_delay(9001, 0.5)
    set_delay(9002, 0.1)
    set_delay(9003, 0.2)
    print("\nDone. Samples are in ml/data.csv")