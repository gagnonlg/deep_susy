import argparse
import logging
import os
import subprocess

import h5py as h5

from deep_susy import dataset, histograms, utils

LOG = logging.getLogger('create_h5')


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('datadir')
    return args.parse_args()


def _main():
    args = _get_args()

    datadir = args.datadir.rstrip('/')
    LOG.info('datadir: %s', datadir)

    masterp = datadir + '/' + os.path.basename(datadir) + '_master.h5'
    LOG.info('master path: %s', masterp)
    dataset.create_master(
        datadir=args.datadir + '/root',
        output=masterp
    )

    splitp = masterp.replace('_master', '')
    LOG.info('split path: %s', splitp)
    dataset.create_split(
        inputp=masterp,
        outputp=splitp,
        default_fractions=(0.5, 0.25, 0.25),
        custom_fractions={
            # 'Diboson': (0, 0, 0),
            'PhHppEG_ttbar': (1, 0, 0),
            'MGPy8EG_ttbar': (0, 1, 0),
            'ttbar': (0, 0, 1)
        }
    )

    LOG.info('Running tests')
    subprocess.check_call([
        'python2',
        utils.top_directory() + '/scripts/test_dataset.py',
        args.datadir + '/root',
        '--master', masterp,
        '--split', splitp
    ])

    masterh5 = h5.File(masterp, 'r')
    bkeys = ['WZjets', 'topEW', 'singletop']
    skeys = ['Gtt_2100_5000_1', 'Gtt_2100_5000_800', 'Gtt_2100_5000_1600']
    ttbar_keypairs = [
        ('PhHppEG_ttbar', False),
        ('MGPy8EG_ttbar', True),
        ('ttbar', True)
    ]
    for ttbar, weighted in ttbar_keypairs:
        plotp = datadir + '/signal_vs_background/' + ttbar
        LOG.info('plot path: %s', plotp)
        histograms.signal_vs_background(
            signals=[masterh5['signal/' + k] for k in skeys],
            backgrounds=[masterh5['background/' + k] for k in bkeys + [ttbar]],
            weighted=weighted,
            directory=plotp
        )


if __name__ == '__main__':
    utils.main(_main, 'create_h5')
