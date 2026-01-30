import logging
import os


def get_logger(out_dir: str, name: str = "smt_cta"):
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(out_dir, "log.txt"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
