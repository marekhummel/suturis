import logging as log
import os
import sys

import suturis.executor
from suturis.config_parser import parse


if __name__ == "__main__":
    io, misc = parse("config.yaml")

    log.info("============ Application start ============")
    if io is not None and misc is not None:
        restart = False

        try:
            suturis.executor.run(io)
            log.info("Main process finished")
        except (KeyboardInterrupt, SystemExit):
            log.info("Suturis was aborted")
        except Exception:
            log.exception("Exception occured, restarting suturis")
            restart = True
        finally:
            suturis.executor.shutdown()

        if restart and misc["automatic_restart"]:
            # Works on linux ?
            os.execv(sys.executable, [sys.executable, __file__] + sys.argv)

    log.info("Application exited")
