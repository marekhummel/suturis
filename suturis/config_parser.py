import logging
import logging.config
import logging.handlers
import os
import os.path
import re
from typing import List

import yaml

from suturis.io.reader import BaseReader
from suturis.io.writer import BaseWriter


def parse(path):
    # Read file
    with open(path) as f:
        config = yaml.safe_load(f.read())

    # Logging
    logging_cfg = config["logging"]
    _config_logging(logging_cfg)

    # IO
    io_config = config["io"]
    io = _define_io(io_config)

    # Misc
    misc_config = config["misc"]

    return io, misc_config


def _config_logging(cfg):
    # Create output dir
    if not os.path.isdir("log"):
        os.mkdir("log")

    # Load logger config
    logging.config.dictConfig(cfg)
    logging.logThreads = False
    logging.logProcesses = False
    logging.logMultiprocessing = False

    # Define namer method to rename files (put date before ext and add counter for exceptions)
    def _namer(new_name):
        sub = re.sub(
            r"\\(?P<name>\w+)\.(?P<ext>\w+)\.(?P<date>[\d_-]+)$",
            r"\\\g<name>.\g<date>.\g<ext>",
            new_name,
        )
        return sub

    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.namer = _namer

    logging.info("Setup of loggers successful")


def _define_io(cfg: dict):
    logging.debug("Define readers and writers")

    # Check input output fields
    inputs: List[dict] = cfg.get("inputs")
    outputs: List[dict] = cfg.get("outputs")
    if inputs is None or outputs is None:
        logging.error("Malformed config: Input or output missing for IO")
        return None

    # Verify input count
    if len(inputs) != 2:
        logging.error("Malformed config: Suturis only works with exactly two inputs")
        return None

    # Create readers and writers
    readers = _create_instances(BaseReader, inputs)
    writers = _create_instances(BaseWriter, outputs)
    return readers, writers if readers is not None and writers is not None else None


def _create_instances(base_class, configs):
    instances = []
    classes = {sc.__name__: sc for sc in base_class.__subclasses__()}
    for i, cfg in enumerate(configs):
        # Get (and remove) type
        cls_name = cfg.pop("type", None)
        if cls_name is None:
            logging.error("Malformed config: IO instances need a type")
            return None

        # Try to find and instantiate class
        cls_obj = classes.get(cls_name)
        if cls_obj is None:
            logging.error(f"Malformed config: Type '{cls_name}' is unknown")
            return None

        try:
            instance = cls_obj(i, **cfg)
            instances.append(instance)
        except TypeError:
            logging.error(
                f"Malformed config: Undefined init params for class '{cls_name}'"
            )
            return None

    return instances
