""" Module to merge and shuffle ROOT ttrees """
import logging
import subprocess

LOGGER = logging.getLogger('dataset.merge')


def merge(paths, output, shuffle_seed=None):
    """ Merge ROOT trees using hadd """

    LOGGER.info('merging %d files into %s', len(paths), output)

    try:
        prog = subprocess.check_output(
            ['which', 'hadd'],
            stderr=subprocess.STDOUT
        )[:-1]  # remove '\n'
        LOGGER.debug('found path to hadd: %s', prog)
    except subprocess.CalledProcessError:
        raise RuntimeError('hadd program not found')

    if len(paths) == 0:
        raise ValueError('input path list is empty')

    args = [prog, '-O', output] + paths
    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            'Merging failed: error in hadd: {}'.format(exc.output)
        )

    if shuffle_seed is not None:
        LOGGER.info('shuffling the merged file')
        LOGGER.debug('rng seed for shuffling: %d', shuffle_seed)
        LOGGER.warning('shuffle not yet implemented')
