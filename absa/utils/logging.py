import os
import logging
import datetime

from absa import log_path


def configure_logging(level=logging.INFO):
    """Logging configuration

    Log formatting.
    Pass logs to terminal and to file.
    """

    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    logging.basicConfig(level=level)
    log_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    log_formatter.formatTime = lambda record, datefmt: datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler(
        f'{log_path}/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
