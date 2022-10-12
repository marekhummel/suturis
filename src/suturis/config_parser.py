import logging
import logging.config
import logging.handlers
import os
import os.path
import re

import yaml

from suturis.io.reader import BaseReader
from suturis.io.writer import BaseWriter
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.processing.computation.preprocessing.base_preprocessor import BasePreprocessor


IOConfig = tuple[list[BaseReader], list[BaseWriter]]
StichingConfig = tuple[list[BasePreprocessor], BaseHomographyHandler, BaseMaskingHandler]
MiscConfig = dict


def parse(path: str) -> tuple[IOConfig | None, StichingConfig | None, MiscConfig]:
    # Read file
    with open(path) as f:
        config = yaml.safe_load(f.read())

    # Logging
    logging_cfg = config["logging"]
    _config_logging(logging_cfg)

    # IO
    io_config = config["io"]
    io = _define_io(io_config)

    # Stitching
    stitching_config = config["stitching"]
    stitching = _define_stitching(stitching_config)

    # Misc
    misc_config = config["misc"]

    # Return
    return io, stitching, misc_config


def _config_logging(cfg: dict) -> None:
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
        if isinstance(handler, logging.handlers.BaseRotatingHandler):
            handler.namer = _namer

    logging.info("Setup of loggers successful")


def _define_io(cfg) -> IOConfig | None:
    logging.debug("Define readers and writers")

    # Check input output fields
    inputs = cfg.get("inputs")
    outputs = cfg.get("outputs")
    if inputs is None or outputs is None:
        logging.error("Malformed config: Input or output missing for IO")
        return None

    # Verify input count
    if len(inputs) != 2:
        logging.error("Malformed config: Suturis only works with exactly two inputs")
        return None

    # Warning if no outputs
    if len(outputs) == 0:
        logging.warning("Config doesn't specify any outputs. Stitching results will be lost.")

    # Create readers and writers
    readers = _create_instances(BaseReader, inputs, True)
    writers = _create_instances(BaseWriter, outputs, True)
    return (readers, writers) if readers is not None and writers is not None else None


def _define_stitching(cfg) -> StichingConfig | None:
    logging.debug("Define stitching classes")

    # Check input output fields
    preprocessors = cfg.get("preprocessing")
    homography = cfg.get("homography")
    masking = cfg.get("masking")
    if homography is None or masking is None:
        logging.error("Malformed config: Homography or Masking missing for Stitching")
        return None

    # Create handlers
    preprocessing_handlers = _create_instances(BasePreprocessor, preprocessors or [], True)
    homography_handler = _create_instances(BaseHomographyHandler, [homography], False)
    masking_handler = _create_instances(BaseMaskingHandler, [masking], False)
    return (
        (preprocessing_handlers, homography_handler[0], masking_handler[0])
        if preprocessing_handlers is not None and homography_handler is not None and masking_handler is not None
        else None
    )


def _create_instances(base_class, configs: list[dict], include_index: bool) -> list | None:
    def _find_subclasses(cls_obj):
        all_subclasses = {}

        for sc in cls_obj.__subclasses__():
            all_subclasses[sc.__name__] = sc
            all_subclasses.update(_find_subclasses(sc))
        return all_subclasses

    instances = []
    classes = _find_subclasses(base_class)

    for i, cfg in enumerate(configs):
        # Get (and remove) type
        cls_name = cfg.pop("type", None)
        if cls_name is None:
            logging.error("Malformed config: Instances need a type")
            return None

        # Try to find and instantiate class
        cls_obj = classes.get(cls_name)
        if cls_obj is None:
            logging.error(f"Malformed config: Type '{cls_name}' is unknown")
            return None

        try:
            instance = cls_obj(i, **cfg) if include_index else cls_obj(**cfg)
            instances.append(instance)
        except TypeError:
            logging.error(f"Malformed config: Undefined init params for class '{cls_name}'")
            return None
        except Exception:
            logging.exception(f"Malformed config: Creation of instance of '{cls_name}' failed")
            return None

    return instances
