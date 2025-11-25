import logging.config
import logging
import os
import inspect
from contextlib import contextmanager


if os.getenv('LOGGING_LEVEL') == "WARNING":
    logging.captureWarnings(True)
if os.getenv("MLFLOW_LOGGING"):
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", 'http://127.0.0.1:5000'))

log_levels = {
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

logger = logging.getLogger('qnbanalytics')
log_level = log_levels.get(os.getenv('LOGGING_LEVEL'), logging.INFO)
logger.setLevel(level=log_level)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(level=log_level)
ch.setFormatter(formatter)
logger.addHandler(ch)


def getLogger():
    return logger


def getArgumentsList(func):
    arg_spec = inspect.getfullargspec(func)
    return arg_spec.args + arg_spec.kwonlyargs

def get_init_parameters(cls):
    return inspect.signature(cls).parameters.keys()

@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
