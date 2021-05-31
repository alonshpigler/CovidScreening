import os
from pathlib import Path


def dict_fields_to_str(d: dict):

    new_d = {}
    for key in d:
        new_d[key] = str(d[key])
    return new_d