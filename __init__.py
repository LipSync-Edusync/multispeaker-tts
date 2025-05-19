import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(
    name="multispeaker_tts",
    log_file="logs/multispeaker_tts.log",
    level=logging.DEBUG,
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

proj_logger = setup_logger()