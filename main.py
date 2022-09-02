import logging as log
import os
import sys

import suturis.executor
from suturis.config_parser import parse

if __name__ == "__main__":
    log.info("Main start")
    io = parse("config.yaml")

    if io is not None:
        try:
            log.info("Process start")
            suturis.executor.run(io)
            log.info("Process finished")
        except KeyboardInterrupt:
            log.info("Suturis was aborted")
        except Exception:
            # Works on linux ?
            log.exception("Exception occured, restarting suturis")
            suturis.executor.shutdown()
            # os.execv(sys.executable, [sys.executable, __file__] + sys.argv)
