""" Module with various utility functions """
# pylint: disable=no-member
import argparse
import itertools
import logging
import os
import subprocess
import sys

import numpy as np

__all__ = ['top_directory', 'main', 'uuid']


def draw_geometrically(vmin, vmax):
    """ Draw integer in log-space """
    return int(draw_exponentially(vmin, vmax))


def draw_exponentially(vmin, vmax):
    """ Draw float in log-space """
    return np.exp(np.random.uniform(np.log(vmin), np.log(vmax)))


def range_sequence(sizes):
    """iterate through a sequence of ranges

    Args:
       sizes: list of range sizes

    Returns: a list of ranges into a larger sequence
             containing subranges of size given by `sizes`
    """
    endpoints = np.cumsum(sizes)
    startpoints = list(itertools.chain([0], endpoints))
    return zip(startpoints, endpoints)


def ensure_suffix(string, suffix, alt=None):

    """ Ensure that a string ends with a suffix

        Args:
          string: the string to verify
          suffix: the suffix to append
          alt: optional, list of valid suffixes

        Returns: the string, with the right suffix
    """

    # Why to set default to []? see:
    # http://stackoverflow.com/questions/9526465/
    if alt is None:
        alt = []

    if not any([string.endswith(s) for s in [suffix] + alt]):
        return string + suffix
    else:
        return string


def get_weights(h5file, dataset):
    """ Return the weight array for the specified dataset in an h5file """
    hdict = {}
    for i, name in enumerate(h5file['header/metadata']):
        hdict[name] = i
    iweight = hdict['M_weight']
    return np.array(h5file[dataset + '/metadata'][:, iweight])


def main(main_function, name):
    """ main function wrapper

    Main function wrapper for cli programs. Will configure the logger,
    log uncaught exceptions and exit with proper return value.

    Args:
      main_function: the function to run
      name: the logger's name
    """

    args = argparse.ArgumentParser()
    args.add_argument('--loglevel', default='INFO')
    args.add_argument('--logfile')
    args, argv = args.parse_known_args()
    sys.argv[1:] = argv

    log_fmt = logging.Formatter(
        '[%(asctime)s %(levelname)s %(module)s.%(funcName)s] %(message)s'
    )
    log_stream = logging.StreamHandler()
    log_stream.setFormatter(log_fmt)
    logging.getLogger().addHandler(log_stream)
    logging.getLogger().setLevel(args.loglevel)
    logging.captureWarnings(True)
    logger = logging.getLogger('main.{}'.format(name))

    if args.logfile is not None:
        path = unique_path(args.logfile)
        log_file = logging.FileHandler(path)
        log_file.setFormatter(log_fmt)
        logging.getLogger().addHandler(log_file)
        logger.info('Copying log to file: %s', path)

    sys.stdout = LoggerWriter(logging.getLogger('stdout'), logging.INFO)

    try:
        logger.info('%s: Begin', str(main_function))
        exit(main_function())
    except SystemExit:
        logger.info('%s: Success', str(main_function))
        raise SystemExit
    except:  # pylint: disable=bare-except
        logger.exception('uncaught exception')
        logger.error('%s: Failure', str(main_function))
        exit(1)


def project_path(path=None):
    """ Return an absolute path in project directory """
    top = top_directory()
    if path is not None:
        return top + '/' + path
    else:
        return top


def top_directory():
    """ Get top directory of code repository """
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.dirname(path)


def unique_path(path):
    """ Return a unique path by appending number """
    unique = path
    i = 1
    while os.path.exists(unique):
        unique = '{}.{}'.format(path, i)
        i += 1
    return unique


def uuid():
    """ generate a uuid string """
    return subprocess.check_output(['uuidgen'])


# pylint: disable=too-few-public-methods
class LoggerWriter(object):
    """ Logger to replace stdout """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        """ alias to stdout """
        self.stdout(message)

    def stdout(self, message):
        """ Write to the log """
        if message != '\n':
            if message[-1] == '\n':
                message = message[:-1]
            self.logger.log(self.level, message)

    def flush(self):
        """ Dummy flush() """
        pass
