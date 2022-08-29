import asyncio

import suturis.executor as runner
import logging as log
import sys


# # asyncio.run(main())
if __name__ == "__main__":
    log.basicConfig(level=log.INFO, handlers=[log.StreamHandler(sys.stdout)])
    log.info("Lets go")
    asyncio.run(runner.run())
