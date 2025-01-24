#!/usr/bin/env python
# Made by: Jonathan Mikler on 2024-11-25

import logging


def configure_logger(logger: logging.Logger) -> logging.Logger:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
    formatter.default_time_format = '%H:%M:%S'
    formatter.default_msec_format = '%s.%03d'
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('/home/iony/DTU/f24/thesis/code/lgvit/lgvit_repo/models/deit_highway/deit.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger