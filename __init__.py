import logging
import os
from logging.handlers import RotatingFileHandler
logging.disable(logging.DEBUG)

def setup_project_logger(
    name="multispeaker_tts",
    log_file="logs/project.log",
    level=logging.DEBUG,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_project_logger()

TRACE = 5
VERBOSE = 15
NOTICE = 25
MSG = 35

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(NOTICE, "NOTICE")
logging.addLevelName(MSG, "MSG")

def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kws)
logging.Logger.trace = trace

def verbose(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kws)
logging.Logger.verbose = verbose

def notice(self, message, *args, **kws):
    if self.isEnabledFor(NOTICE):
        self._log(NOTICE, message, args, **kws)
logging.Logger.notice = notice

def msg(self, message, *args, **kws):
    if self.isEnabledFor(MSG):
        self._log(MSG, message, args, **kws)
logging.Logger.msg = msg


