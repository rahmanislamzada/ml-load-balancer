"""Slow backend: ~0.5 s delay."""
import asyncio
from backends.base_backend import Backend

backend = Backend("slow", delay=0.5)


async def start_server(port: int = 9001):
    await backend.start(port)


if __name__ == "__main__":
    import sys
    asyncio.run(start_server(int(sys.argv[1]) if len(sys.argv) > 1 else 9001))