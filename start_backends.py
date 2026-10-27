"""Start all 3 backends (slow 9001, fast 9002, variable 9003) in one process."""
import asyncio
from backends.slow_backend import start_server as start_slow
from backends.fast_backend import start_server as start_fast
from backends.variable_backend import start_server as start_variable


async def main():
    await asyncio.gather(start_slow(9001), start_fast(9002), start_variable(9003))


if __name__ == "__main__":
    asyncio.run(main())