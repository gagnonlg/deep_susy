""" Module with various utility functions """

import argparse
import logging
import os
import subprocess
import sys


__all__ = ['top_directory', 'main', 'uuid']


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
    args, argv = args.parse_known_args()
    sys.argv[1:] = argv

    logging.basicConfig(
        level=args.loglevel,
        format='[%(name)s] %(levelname)s %(message)s'
    )

    try:
        exit(main_function())
    except SystemExit:
        raise
    except:  # pylint: disable=bare-except
        logger = logging.getLogger('main.{}'.format(name))
        logger.exception('uncaught exception')
        exit(1)


def top_directory():
    """ Get top directory of code repository """
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.dirname(path)


def uuid():
    """ generate a uuid string """
    return subprocess.check_output(['uuidgen'])
