import asyncio
import logging as log
import os
import sys

import suturis.executor
from suturis.config_parser import parse

if __name__ == "__main__":
    io = parse("config.yaml")
    if io is not None:
        try:
            log.info("Process start")
            asyncio.run(suturis.executor.run(io))
            log.info("Process finished")
        except KeyboardInterrupt:
            log.info("Suturis was aborted")
        except:
            # Works on linux ?
            log.exception("Exception occured, restarting suturis")
            suturis.executor.shutdown()
            os.execv(sys.executable, [sys.executable, __file__] + sys.argv)
