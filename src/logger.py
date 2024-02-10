"""
Logging Functions

These module defines the logging mechanism.
"""

import logging

LOGGER_NAME = 'GALAX-IA'

def setup_logger(logger_name: str = LOGGER_NAME,
                 file_name: str = None,
                 level_terminal=logging.DEBUG,
                 level_file=logging.DEBUG,
                 mode='w') -> logging.Logger:

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(level_terminal)
    formatter = logging.Formatter('%(asctime)s\t%(message)s',
                                  datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if file_name is not None:
        try:
            fh = logging.FileHandler(file_name, mode=mode)
            fh.setLevel(level_file)
            formatter = logging.Formatter(
                '%(asctime)s\t%(name)s\t%(levelname)s: %(message)s',
                datefmt='%Y/%m/%d %H:%M:%S')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.error(f"Error creating file handler: {e}")

    return logger


def get_logger(module_name: str,
               logger_name: str = LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(logger_name).getChild(module_name)
