import logging
import time


def get_logger(log_path):
    log_name = time.strftime("%Y%m%d%H%M%S", time.localtime())
    logger = logging.getLogger(log_name)
    file_handler = logging.FileHandler(log_path)
    stdout_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.INFO)

    return logger
