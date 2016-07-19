""" Module with various utility functions """

import os

__all__ = ['top_directory']


def top_directory():
    """ Get top directory of code repository """
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.dirname(path)
