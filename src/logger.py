import logging
import sys

def init_logger(name=__name__, level=logging.DEBUG):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s --- %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger