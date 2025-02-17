#!/usr/bin/env python
# Made by: Jonathan Mikler on 2024-11-25

import torch
import logging
from typing import Dict
from colorama import Fore, Style


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


def yellow_txt(txt: str) -> str:
    return f"{Fore.YELLOW}{txt}{Style.RESET_ALL}"

def print_dict(dictionary, ident="", braces=1):
    """Recursively prints nested dictionaries."""

    for key, value in dictionary.items():
        if isinstance(value, dict):
            print(f'{ident}{braces*"["}{key}{braces*"]"}')
            print_dict(value, ident + "  ", braces + 1)
        else:
            print(f"{ident}{key} = {value}")

def get_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    return {
        'shape': tensor.shape,
        'sum': tensor.sum().item(),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'norm': torch.norm(tensor).item(),
        # 'hash': hash(tensor.cpu().numpy().tobytes())
    }

def gts(tensor):
    print_dict(get_tensor_stats(tensor))