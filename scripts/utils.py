""" Module with various utility functions """

import logging
import os
import subprocess

__all__ = ['top_directory', 'main']


def top_directory():
    """ Get top directory of code repository """
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.dirname(path)


def main(main_function, name, loglevel=logging.INFO):
    """ main function wrapper

    Main function wrapper for cli programs. Will configure the logger,
    log uncaught exceptions and exit with proper return value.

    Args:
      main_function: the function to run
      name: the logger's name
    """

    logging.basicConfig(
        level=loglevel,
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


def uuid():
    """ generate a uuid string """
    return subprocess.check_output(['uuidgen'])
