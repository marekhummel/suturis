import logging as log
import logging.config
import logging.handlers
import os
import os.path
import re
from typing import List, Tuple

import yaml

from suturis.io.reader import BaseReader, FileReader
from suturis.io.writer import BaseWriter, ScreenOutput


def parse(path):
    # Read file
    with open(path) as f:
        config = yaml.safe_load(f.read())

    # Logging
    logging_cfg = config["logging"]
    _config_logging(logging_cfg)

    # IO
    io_config = config["io"]
    return _define_io(io_config)


def _config_logging(cfg):
    # Create output dir
    if not os.path.isdir("log"):
        os.mkdir("log")

    # Load logger config
    logging.config.dictConfig(cfg)

    # Define namer method to rename files (put date before ext and add counter for exceptions)
    def _namer(new_name):
        sub = re.sub(
            r"\\(?P<name>\w+)\.(?P<ext>\w+)\.(?P<date>[\d_-]+)$",
            r"\\\g<name>.\g<date>.\g<ext>",
            new_name,
        )
        return sub

    for handler in log.getLogger().handlers:
        handler.namer = _namer


def _define_io(cfg: dict):
    log.debug("Define readers and writers")

    # Check input output fields
    inputs: List[dict] = cfg.get("inputs", None)
    outputs: List[dict] = cfg.get("outputs", None)
    if inputs is None or outputs is None:
        log.error("Malformed config: Input or output missing for IO")
        return None

    # Verify input count
    if len(inputs) != 2:
        log.error("Malformed config: Suturis only works with exactly two inputs")
        return None

    # Create readers and writers
    readers = _create_instances(BaseReader, inputs)
    writers = _create_instances(BaseWriter, outputs)
    if readers is None or writers is None:
        return None

    return readers, writers


def _create_instances(base_class, configs):
    instances = []
    classes = {sc.__name__: sc for sc in base_class.__subclasses__()}
    for cfg in configs:
        # Get (and remove) type
        cls_name = cfg.pop("type", None)
        if cls_name is None:
            log.error("Malformed config: IO instances need a type")
            return None

        # Try to find and instantiate class
        cls_obj = classes.get(cls_name, None)
        if cls_obj is None:
            log.error(f"Malformed config: Type '{cls_name}' is unknown")
            return None

        try:
            instance = cls_obj(**cfg)
            instances.append(instance)
        except TypeError:
            log.error(f"Malformed config: Undefined init params for class '{cls_name}'")
            return None

    return instances
