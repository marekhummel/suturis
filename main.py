import asyncio
import logging as log
import logging.config
import os
import os.path

import yaml
import suturis.processing.computation.manager as mgr

import suturis.executor


if __name__ == "__main__":
    if not os.path.isdir("log"):
        os.mkdir("log")
    with open("logconf.yaml") as f:
        logging.config.dictConfig(yaml.safe_load(f.read()))

    while True:
        try:
            log.info("Process start")
            asyncio.run(suturis.executor.run())
            log.info("Process finished")
        except:
            log.exception("Exception occured, restarting suturis")
            mgr.local_params = None
        else:
            break
