from os.path import exists
from os import makedirs


def path_exists(path):
    if not exists(path):
        makedirs(path)
    return path
