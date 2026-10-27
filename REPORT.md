# Technical Report: ML-Based Adaptive Load Balancer

## 1. Goal

Build a load balancer at the socket level to demonstrate core networking concepts (TCP, HTTP parsing, timeouts, concurrency, health checking), then test whether a machine-learning model can route requests better than the classic round-robin and least-connections strategies, especially when backend performance changes at runtime.

## 2. System overview

| Component | Role | Port |
|---|---|---|
| Backends (slow, fast, variable) | Simulated HTTP servers with configurable delay | 9001, 9002, 9003 |
| Proxy / load balancer | Accepts clients, selects a backend, forwards and relays | 8080 |
| Strategies | Round-robin, least-connections, random (data collection), ML | n/a |
| Health checker | Periodic `GET /health` per backend | n/a |

### Request lifecycle

1. A client opens a TCP connection to the proxy.
2. The proxy reads until `\r\n\r\n`, parses the request line and headers manually, and reads the body if `Content-Length` is present.
3. It removes hop-by-hop headers, adds `X-Forwarded-For` and forces `Connection: close` towards the backend.
4. The strategy selects a backend among the healthy ones that have not been tried yet.
5. The proxy opens a **new** TCP connection to that backend, forwards the request and reads until EOF.
6. The response is relayed to the client. Latency and backend state are updated.
7. On timeout, connection error or empty reply, the proxy retries on another backend. If none remain it returns `502 Bad Gateway`.

## 3. Networking concepts and where they appear

| Concept | Where | Layer |
|---|---|---|
| TCP connection setup, ports, sockets | Proxy listens on 8080 and opens client and backend connections | Transport (L4) |
| Manual HTTP/1.1 parsing (request line, headers, `Content-Length`) | `read_request` in `proxy.py` | Application (L7) |
| Layer-7 routing decision | Proxy inspects HTTP before choosing a backend | Application (L7) |
| `X-Forwarded-For`, hop-by-hop header handling | Request rewriting | Application (L7) |
| `Content-Length` framing, `Connection: close` | Backends and proxy mark message boundaries | Application (L7) |
| Timeouts (header 5 s, connect 3 s, read 10 s) | `asyncio.wait_for` around every network wait | Transport / Application |
| Concurrency with an event loop | `asyncio` handles many connections in one thread | n/a |
| Health checking | Periodic HTTP `GET /health` | Application (L7) |
| Failure handling and retry | Proxy retries on another backend, returns 502 if all fail | Application (L7) |

Design note: the proxy uses a new backend connection per request with `Connection: close`. This keeps framing simple and robust (the end of the response is the end of the connection) at the cost of connection setup overhead. Connection pooling and keep-alive would be a natural improvement.

## 4. Strategies

- **Round-robin**: cycles through healthy backends. Ignores speed and load.
- **Least-connections**: picks the backend with the fewest in-flight requests, tie-broken by average latency.
- **Random**: used only to collect diverse training data.
- **ML**: predicts latency per backend and picks the lowest prediction; with probability 5% it explores a random backend (epsilon-greedy); falls back to least-connections if no model is available or prediction fails.

All strategies implement the same `Strategy` interface (`select_backend`), so the proxy treats them interchangeably.

## 5. Machine-learning approach

### Data collection

The proxy runs with the `random` strategy and `--log-data`. For each request it records a snapshot of the chosen backend's state **at decision time** and the **real measured latency**. `collect_data.py` varies backend delays (six configurations, including swapping fast and slow) and load (three concurrency levels), so the model sees diverse conditions. Result: 3,300 samples.

### Features (same order in training and in the proxy)

| Feature | Meaning |
|---|---|
| `active_connections` | In-flight requests on the backend |
| `avg_latency` | Mean of the last 20 successful latencies |
| `error_rate` | Exponential moving average of failures |
| `request_rate` | Requests sent to the backend per second over 10 s |

### Model

`GradientBoostingRegressor` (100 trees, depth 3), deliberately small because prediction runs on every request. 80/20 train/test split.

### Evaluation

| Metric | Value |
|---|---|
| Model MAE | 75.4 ms |
| Model RMSE | 123.8 ms |
| R² | 0.743 |
| Baseline: always predict the mean | MAE 213.5 ms |
| Baseline: use `avg_latency` directly | MAE 73.4 ms |

Feature importance: `avg_latency` 0.962, `request_rate` 0.025, `active_connections` 0.014, `error_rate` 0.000.

### Honest interpretation

The model is far better than predicting the mean, but **not better than the trivial baseline of using the recent average latency**. Feature importance confirms this: the model mostly learned "recent latency predicts next latency". This is expected in the simulation because a backend's response time depends almost entirely on its own configured delay, and the asynchronous simulated servers do not slow down under load, so `active_connections` carries little signal. `error_rate` is zero because no backend failed during data collection.

What the ML strategy adds in practice is a smooth, learned combination of signals and a mechanism (epsilon-greedy) to keep exploring backends that look slow.

## 6. Benchmark

### Method

- Three scenarios: **steady** (300 requests, concurrency 20), **burst** (100 at concurrency 5, then 300 at 40, then 100 at 5), **slowdown** (200 requests, then the `fast` backend changes from 0.1 s to 0.8 s and 400 more requests follow).
- Each (scenario, strategy) combination is repeated 3 times, interleaved to reduce time drift; the benchmark script starts and stops the proxy for every run and resets backend delays.
- Metrics: average, p95 and p99 latency, throughput, error rate, and share of requests sent to `fast`.
- Closed-loop load generator with fixed concurrency, so throughput rises when latency falls.

### Results (mean of 3 repetitions)

| Scenario / phase | Strategy | Avg (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Fast share |
|---|---|---|---|---|---|---|
| steady | rr | 373.2 | 722.8 | 788.0 | 50.9 | 33.3% |
| steady | lc | 229.3 | 604.7 | 753.5 | 79.7 | 69.2% |
| steady | ml | 200.1 | 615.8 | 685.0 | 88.5 | 79.0% |
| burst / high | rr | 379.0 | 737.0 | 795.4 | 92.4 | 33.3% |
| burst / high | lc | 238.4 | 607.1 | 749.9 | 136.7 | 67.6% |
| burst / high | ml | 135.0 | 144.7 | 510.6 | 215.2 | 97.7% |
| slowdown / before | rr | 371.3 | 709.2 | 773.0 | 49.3 | 33.5% |
| slowdown / before | lc | 230.8 | 602.0 | 732.5 | 76.6 | 69.0% |
| slowdown / before | ml | 239.9 | 555.7 | 598.2 | 90.3 | 70.3% |
| slowdown / after | rr | 605.0 | 820.4 | 825.6 | 31.8 | 33.2% |
| slowdown / after | lc | 572.0 | 819.3 | 827.2 | 33.8 | 21.5% |
| slowdown / after | ml | 534.7 | 816.7 | 827.5 | 36.2 | 7.4% |

All runs finished with 0% errors. Raw data: `results/raw.csv`; aggregates: `results/summary.csv`.

### Discussion

- **Round-robin** is worst whenever backends differ, because it sends one third of the traffic to the slowest servers.
- **Least-connections** adapts to load and is a strong baseline.
- **ML** is the best or tied-best in most phases. The large burst advantage (p95 145 ms vs 607 ms) comes from sending about 98% of traffic to the fastest backend.
- **Slowdown**: round-robin does not react (33% to `fast` both before and after). Least-connections reduces the share to 21.5%, ML to 7.4%, so ML reacts faster to the latency increase. However, after the slowdown all backends take roughly 0.5 to 0.8 s, so there is little room to improve and the latency gap is small (535 vs 572 vs 605 ms) and p95/p99 are almost identical.
- **Epsilon-greedy cost**: in steady state ML's p95 (616 ms) is slightly above least-connections (605 ms) because 5% of requests are deliberately routed to a random backend.

## 7. Limitations

1. **The burst advantage is partly a simulation artifact.** Simulated backends have no capacity limit, so concentrating about 98% of traffic on one backend does not slow it down. A real server would saturate, and a purely greedy predictor could overload it. A capacity-aware model or a load cap per backend would be required.
2. **The model is close to a simple rule.** It does not beat the `avg_latency` baseline on MAE, so the ML contribution is modest in this environment.
3. **Single machine, loopback network.** No real network latency, jitter or loss; results are not representative of a production deployment.
4. **Limited statistics.** Three repetitions per condition; no confidence intervals were computed, so small differences (for example ML vs least-connections in steady state) should not be over-interpreted.
5. **Training distribution.** The model was trained on six delay configurations and may extrapolate poorly to very different conditions; retraining would be needed when traffic or backends change.
6. **Protocol scope.** HTTP/1.1 with `Connection: close` only: no keep-alive, TLS, chunked request bodies, WebSockets or HTTP/2.
7. **Failure data.** `error_rate` never varied during training, so the model has not learned how failures affect latency; failover relies on health checks and retry, not on ML.

## 8. Future work

- Online learning or periodic retraining from the proxy's own logs.
- Capacity-aware routing (per-backend concurrency caps) and a reinforcement-learning or bandit formulation instead of supervised regression.
- Keep-alive and connection pooling towards backends.
- Real backends and network emulation (`tc netem`, Mininet) for latency and loss.
- TLS termination and metrics export (Prometheus).
- More repetitions with confidence intervals.

## 9. Design FAQ

**Why TCP and `asyncio` instead of threads?** HTTP needs reliable, ordered delivery, which TCP provides. An event loop serves many connections in one thread and yields while waiting on I/O.

**Why `Connection: close` towards backends?** The end of the response is signalled by the connection closing, which avoids implementing response framing for chunked encoding and keep-alive. The trade-off is extra connection setup per request.

**Why is it a layer-7 balancer?** It parses HTTP and could route on path or headers, while a layer-4 balancer would only forward TCP streams.

**What happens if a backend crashes mid-test?** The next request to it fails, the proxy retries on another backend, and the health checker marks the backend DOWN within a couple of seconds and brings it back when `/health` succeeds.

**Why can ML be worse than the simplest rule here?** Because backend latency is dominated by the backend's own speed; a model that mostly uses recent latency cannot beat a rule that uses recent latency. Its value shows up only where there are more signals to combine.

**Why does the ML strategy fall back to least-connections?** So the system never fails because of the model: a missing model file or a failed prediction degrades to a sane heuristic.