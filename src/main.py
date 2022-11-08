import logging as log
import os
import sys
import argparse

import suturis.executor
from suturis.config_parser import parse


if __name__ == "__main__":
    # Read and parse config
    parser = argparse.ArgumentParser(description="Suturis - real time image stitching.")
    parser.add_argument("config", type=str, nargs="?", default="src/config.yaml", help="Path to yaml config file")
    args = parser.parse_args()
    # args.config = "data/examples/bow_stern.yaml"
    io, stitching, misc = parse(args.config)

    # Start application
    if io and stitching and misc:
        log.info("============ Application started ============")
        restart = False

        try:
            suturis.executor.run(io, stitching, misc)
        except (KeyboardInterrupt, SystemExit):
            log.info("Suturis was aborted")
        except Exception:
            log.exception("Exception occured, restarting suturis")
            restart = True
        finally:
            suturis.executor.shutdown()
            log.info("Main process finished")

        if restart and misc["automatic_restart"]:
            # Works on linux ?
            os.execv(sys.executable, [__file__] + sys.argv)

    log.info("============ Application exited ============")
