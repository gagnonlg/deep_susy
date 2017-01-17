""" Module to merge and shuffle ROOT ttrees """
import logging
import os.path
import subprocess

import utils

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
        LOGGER.error('Error in hadd: %s', exc.output)
        raise RuntimeError('Merging failed')

    if shuffle_seed is not None:
        LOGGER.info('shuffling the merged file')
        LOGGER.debug('rng seed for shuffling: %d', shuffle_seed)

        prog = '{}/bin/shuffle_tree'.format(utils.top_directory())
        if not os.path.exists(prog):
            raise RuntimeError('{} not found'.format(prog))
        else:
            LOGGER.debug('found path to shuffle_tree: %s', prog)

        tmp = '{}.root'.format(utils.uuid())
        try:
            subprocess.check_call([prog, str(shuffle_seed), output, tmp])
            subprocess.check_call(
                ['mv', tmp, output],
                stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('Shuffling failed: {}'.format(exc.output))
        finally:
            subprocess.check_call(['rm', '-f', tmp])
