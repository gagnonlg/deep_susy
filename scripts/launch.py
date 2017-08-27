import argparse
import logging
import subprocess
import os
import time

import h5py as h5
import numpy as np
import theano

import utils


def main():

    # parse the args
    args = argparse.ArgumentParser()
    args.add_argument('--data', required=True)
    args.add_argument('--definition', required=True)
    args.add_argument('--logjobs', default='/dev/stdout')
    args.add_argument('--debug', action='store_true', default=False)
    grp = args.add_mutually_exclusive_group(required=True)
    grp.add_argument('--train', action='store_true')
    grp.add_argument('--optimize', type=int)
    args = args.parse_args()

    if args.train:
        train(args.definition, args.data)
    else:
        optimize(
            args.optimize,
            args.data,
            args.definition,
            args.logjobs,
            args.debug
        )


def optimize(ntries, data, defn, logpath, debug):

    with open(logpath, 'w') as logf:
        for _ in range(ntries):
            logf.write('{}\n'.format(launch(data, defn, debug)))
            time.sleep(0.5)


def launch(data, defn, debug=False):

    cmd = [
        'qsub',
        '-d', '/lcg/storage15/atlas/gagnon/work',
        '-N', 'optimization',
        '-joe',
    ]

    qsub = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    jobid, _ = qsub.communicate(job_script(data, defn, debug))

    return jobid.strip()


def train(defpath, datapath):

    log = logging.getLogger(__name__)

    log.debug('HOSTNAME: %s', os.getenv('HOSTNAME'))
    log.debug('theano.config.gcc.cxxflags=%s', theano.config.gcc.cxxflags)

    # load the definition
    defn = {}
    execfile(defpath, defn)
    if 'definition' not in defn:
        raise RuntimeError('definition not found in file ' + defpath)
    definition = defn['definition']

    # load the data
    data = h5.File(datapath, 'r')
    trainX = np.array(data['training/inputs'])
    trainY = np.array(data['training/labels'])
    testX = np.array(data['validation/inputs'])
    testY = np.array(data['validation/labels'])

    # train and eval
    definition.log()
    definition.save()
    trained = definition.train(trainX, trainY)
    trained.save()
    trained.evaluate(testX, testY).save()


def job_script(datapath, defpath, debug=False):
    if debug:
        debug = '--loglevel=DEBUG'
    else:
        debug = ''
    return """
mkdir ${PBS_JOBID}_${PBS_JOBNAME}
cd ${PBS_JOBID}_${PBS_JOBNAME}

git clone ~/dev/deep_susy/git .
. scripts/setup.sh

scripts/theano_wrapper python2 -u scripts/launch.py --train --data %s --definition %s %s |& tee launch.log
""" % (os.path.abspath(datapath), os.path.abspath(defpath), debug)  # noqa


if __name__ == '__main__':
    utils.main(main, 'launch')
