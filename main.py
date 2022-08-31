import asyncio
import logging as log
import logging.config
import logging.handlers
import os
import os.path
import re

import yaml
import suturis.processing.computation.manager as mgr

import suturis.executor


def config_logging():
    # Create output dir
    if not os.path.isdir("log"):
        os.mkdir("log")

    # Load logger config
    with open("logconf.yaml") as f:
        logging.config.dictConfig(yaml.safe_load(f.read()))

    # Define namer to rename files (put date before ext and add counter for exceptions)
    def _namer(new_name):
        sub = re.sub(
            r"\\(?P<name>\w+)\.(?P<ext>\w+)\.(?P<date>[\d_-]+)$",
            r"\\\g<name>.\g<date>.\g<ext>",
            new_name,
        )
        return sub

    for handler in log.getLogger().handlers:
        handler.namer = _namer


if __name__ == "__main__":
    config_logging()

    while True:
        try:
            log.info("Process start")
            asyncio.run(suturis.executor.run())
            log.info("Process finished")
        except KeyboardInterrupt:
            log.info("Suturis was aborted")
        except:
            log.exception("Exception occured, restarting suturis")
            mgr.local_params = None
        else:
            break
