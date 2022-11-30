import logging


def get_logger(log_path, save=True, display=True):
    logger = logging.getLogger()
    level = logging.INFO
    logger.setLevel(level)
    if save:
        info_file_handler = logging.FileHandler(log_path, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if display:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger
