import logging
import os
from logging.handlers import RotatingFileHandler

TRACE = 5
VERBOSE = 15
NOTICE = 25
MSG = 35
MSG2 = 45 # train_speaker_encoder.py

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(NOTICE, "NOTICE")
logging.addLevelName(MSG, "MSG")
logging.addLevelName(MSG2, "MSG2")

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

def msg2(self, message, *args, **kws):
    """
    Logs a message with level MSG2, including an iteration count.

    Args:
        self: The logger instance.
        message: The message to log.
        *args:  Additional positional arguments for the message.
    """
    if self.isEnabledFor(MSG2):
        # Ensure the counter exists.  If not, initialize it.
        if not hasattr(self, '_msg2_call_count'):
            self._msg2_call_count = 0
        self._msg2_call_count += 1  # Increment the counter

        # Include the iteration count in the message.
        new_message = f"MSG2 Call #{self._msg2_call_count}: {message}"
        self._log(MSG2, new_message, args, **kws)
logging.Logger.msg2 = msg2

# Disable loggings
# logging.disable(logging.DEBUG)

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
    
    logger.info(" ==================== Logger initialized ===================== ")

    return logger

logger = setup_project_logger()



