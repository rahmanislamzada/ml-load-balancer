"""Variable backend: unstable delay, random between 0.2 s and 0.8 s per request."""
import asyncio
from backends.base_backend import Backend

# delay = 0.2 s base + random 0..0.6 s  ->  each request takes 0.2-0.8 s
backend = Backend("variable", delay=0.2, jitter=0.6)


async def start_server(port: int = 9003):
    await backend.start(port)


if __name__ == "__main__":
    import sys
    asyncio.run(start_server(int(sys.argv[1]) if len(sys.argv) > 1 else 9003))