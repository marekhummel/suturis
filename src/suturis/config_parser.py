import logging
import logging.config
import logging.handlers
import os
import os.path
import re
from typing import TypeVar

import yaml

from suturis.io.reader import BaseReader
from suturis.io.writer import BaseWriter
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.processing.computation.postprocessing import BasePostprocessor
from suturis.processing.computation.preprocessing import BasePreprocessor


IOConfig = tuple[list[BaseReader], list[BaseWriter]]
StichingConfig = tuple[list[BasePreprocessor], BaseHomographyHandler, BaseMaskingHandler, list[BasePostprocessor]]
MiscConfig = dict


T = TypeVar("T")


def parse(path: str) -> tuple[IOConfig | None, StichingConfig | None, MiscConfig]:
    """Parses the YAML config at the given path to required objects.

    Parameters
    ----------
    path : str
        Path to YAML config

    Returns
    -------
    tuple[IOConfig | None, StichingConfig | None, MiscConfig]
        Created objects based on config
    """
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
    """Parses the configuration for the logging and applies it. Schema controlled by pythons internal logging.

    Parameters
    ----------
    cfg : dict
        The config section for logging
    """
    # Create output dir
    if not os.path.isdir("log"):
        os.mkdir("log")

    # Load logger config
    logging.config.dictConfig(cfg)
    logging.logThreads = False
    logging.logProcesses = False
    logging.logMultiprocessing = False

    # Define namer method to rename files (put date before ext and add counter for exceptions)
    def _namer(new_name: str) -> str:
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


def _define_io(cfg: dict) -> IOConfig | None:
    """Parses config section for IO (readers and writers)

    Parameters
    ----------
    cfg : dict
        Config for IO

    Returns
    -------
    IOConfig | None
        Readers and writers or None if an error occured.
    """
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
    readers = _create_instances(BaseReader, inputs)
    writers = _create_instances(BaseWriter, outputs)
    return (readers, writers) if readers is not None and writers is not None else None


def _define_stitching(cfg: dict) -> StichingConfig | None:
    """Parses config section for stitching (preprocessors, homography and mask handler).

    Parameters
    ----------
    cfg : dict
        Config for Stitching

    Returns
    -------
    StichingConfig | None
        Preprocessing, homography and masking handler(s) or None, if an error occured.
    """
    logging.debug("Define stitching classes")

    # Check input output fields
    preprocessors = cfg.get("preprocessing")
    homography = cfg.get("homography")
    masking = cfg.get("masking")
    postprocessors = cfg.get("postprocessing")
    if homography is None or masking is None:
        logging.error("Malformed config: Homography or Masking missing for Stitching")
        return None

    # Create handlers
    preprocessing_handlers = _create_instances(BasePreprocessor, preprocessors or [])
    homography_handler = _create_instance(BaseHomographyHandler, homography)
    masking_handler = _create_instance(BaseMaskingHandler, masking)
    postprocessing_handlers = _create_instances(BasePostprocessor, postprocessors or [])

    if (
        preprocessing_handlers is None
        or homography_handler is None
        or masking_handler is None
        or postprocessing_handlers is None
    ):
        return None

    if any(post._caching_enabled for post in postprocessing_handlers) and not masking_handler._caching_enabled:
        logging.warning("Some post processors have caching enabled, while the masking handler has not")

    if masking_handler._caching_enabled and not homography_handler._caching_enabled:
        logging.warning("The masking handler has caching enabled, while the homography handler has not")

    return preprocessing_handlers, homography_handler, masking_handler, postprocessing_handlers


def _create_instances(base_class: type[T], configs: list[dict]) -> list[T] | None:
    """Creates subclass instances based on a given base class and their config.

    Parameters
    ----------
    base_class : type
        Defined classes in the configs have to derive from this class
    configs : list[dict]
        Configs for each instance to be created. Needs to have a "type" and all required params that are used in the
        respective constructor.
    include_index : bool
        If set, the list index will be provided to the instance as well

    Returns
    -------
    list | None
        List of instances or None, if any failed.
    """

    instances = []
    classes = _find_subclasses(base_class)

    for i, cfg in enumerate(configs):
        instance = _create_instance(base_class, cfg, classes, i)
        if instance is None:
            return None

        instances.append(instance)

    return instances


def _create_instance(
    base_class: type[T], cfg: dict, subclasses: dict[str, type[T]] | None = None, index: int | None = None
) -> T | None:
    if subclasses is None:
        subclasses = _find_subclasses(base_class)

    # Get (and remove) type
    cls_name = cfg.pop("type", None)
    if cls_name is None:
        logging.error("Malformed config: Instances need a type")
        return None

    # Try to find and instantiate class
    cls_obj = subclasses.get(cls_name)
    if cls_obj is None:
        logging.error(f"Malformed config: type '{cls_name}' is unknown")
        return None

    try:
        # Difficult to type since there is no base class for all configurable classes
        instance = cls_obj(index, **cfg) if index else cls_obj(**cfg)  # type: ignore
        return instance
    except TypeError:
        logging.error(f"Malformed config: Undefined init params for class '{cls_name}'")
        return None
    except Exception:
        logging.exception(f"Malformed config: Creation of instance of '{cls_name}' failed")
        return None


def _find_subclasses(cls_obj: type) -> dict[str, type]:
    all_subclasses = {}

    for sc in cls_obj.__subclasses__():
        all_subclasses[sc.__name__] = sc
        all_subclasses |= _find_subclasses(sc)
    return all_subclasses
