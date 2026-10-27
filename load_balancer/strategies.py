from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Optional

FEATURE_NAMES = ["active_connections", "avg_latency", "error_rate", "request_rate"]
DEFAULT_MODEL_PATH = os.path.join("ml", "model.joblib")
RATE_WINDOW = 10.0  # seconds used to compute request_rate


class BackendInfo:

    def __init__(self, host: str, port: int, name: str):
        self.host = host
        self.port = port
        self.name = name
        self.is_healthy = True
        self.active_connections = 0
        self.error_rate: float = 0.0
        self.last_check: float = time.time()
        self.avg_latency: float = 0.0
        self.recent_latencies: List[float] = []

    def __repr__(self) -> str:
        return (f"BackendInfo({self.name}, {self.host}:{self.port}, "
                f"healthy={self.is_healthy}, conn={self.active_connections})")


class Strategy(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def select_backend(self, backends: Dict[str, BackendInfo]) -> Optional[BackendInfo]:
        """Pick one backend from the given candidates (or None)."""

    def update_backend_state(self, backend_name: str, info: BackendInfo) -> None:
        """Hook called after a request finishes. Optional."""


def _healthy(backends: Dict[str, BackendInfo]) -> List[BackendInfo]:
    return [b for b in backends.values() if b.is_healthy]


def _least_connections(candidates: List[BackendInfo]) -> BackendInfo:
    # tie-breaker: lower average latency wins
    return min(candidates, key=lambda b: (b.active_connections, b.avg_latency))


class RoundRobinStrategy(Strategy):

    def __init__(self) -> None:
        self._index = 0

    @property
    def name(self) -> str:
        return "round-robin"

    async def select_backend(self, backends):
        healthy = _healthy(backends)
        if not healthy:
            return None
        choice = healthy[self._index % len(healthy)]
        self._index += 1
        return choice


class LeastConnectionsStrategy(Strategy):

    @property
    def name(self) -> str:
        return "least-connections"

    async def select_backend(self, backends):
        healthy = _healthy(backends)
        return _least_connections(healthy) if healthy else None


class RandomStrategy(Strategy):

    @property
    def name(self) -> str:
        return "random"

    async def select_backend(self, backends):
        healthy = _healthy(backends)
        return random.choice(healthy) if healthy else None


class MLStrategy(Strategy):

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, epsilon: float = 0.05):
        self.model_path = model_path
        self.epsilon = epsilon      # probability of exploring a random backend
        self.model = None
        self._recent: Dict[str, Deque[float]] = {}
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            print(f"[ML] no trained model at '{self.model_path}' -> using least-connections")
            return
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            print(f"[ML] loaded trained model from '{self.model_path}'")
        except Exception as e:
            print(f"[ML] could not load model ({e!r}) -> using least-connections")

    @property
    def name(self) -> str:
        return "ml-based" if self.model is not None else "ml-based (fallback: least-connections)"

    def _request_rate(self, backend_name: str, now: float) -> float:
        q = self._recent.setdefault(backend_name, deque())
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
        return len(q) / RATE_WINDOW

    def build_features(self, b: BackendInfo, now: float) -> List[float]:
        """Order must match FEATURE_NAMES."""
        return [float(b.active_connections), b.avg_latency, b.error_rate,
                self._request_rate(b.name, now)]

    async def select_backend(self, backends):
        healthy = _healthy(backends)
        if not healthy:
            return None
        now = time.time()
        choice = None
        if self.model is not None:
            if random.random() < self.epsilon:
                choice = random.choice(healthy)              # exploration
            else:
                try:
                    rows = [self.build_features(b, now) for b in healthy]
                    predictions = self.model.predict(rows)   # exploitation
                    choice = healthy[min(range(len(healthy)), key=lambda i: predictions[i])]
                except Exception as e:
                    print(f"[ML] prediction failed ({e!r}) -> least-connections")
        if choice is None:
            choice = _least_connections(healthy)
        self._recent.setdefault(choice.name, deque()).append(now)
        return choice