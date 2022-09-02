import asyncio
import logging as log
from suturis.config_parser import parse

import suturis.processing.computation.manager as mgr

import suturis.executor


if __name__ == "__main__":
    io = parse("config.yaml")
    if io is not None:
        while True:
            try:
                log.info("Process start")
                asyncio.run(suturis.executor.run(io))
                log.info("Process finished")
            except KeyboardInterrupt:
                log.info("Suturis was aborted")
            except:
                log.exception("Exception occured, restarting suturis")
                mgr.local_params = None
            else:
                break
