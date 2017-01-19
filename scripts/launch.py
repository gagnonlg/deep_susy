import argparse
import subprocess
import os
import time

import h5py as h5
import numpy as np

import utils


def main():

    # parse the args
    args = argparse.ArgumentParser()
    args.add_argument('--data', required=True)
    args.add_argument('--definition', required=True)
    grp = args.add_mutually_exclusive_group(required=True)
    grp.add_argument('--train', action='store_true')
    grp.add_argument('--optimize', type=int)
    args = args.parse_args()

    if args.train:
        train(args.definition, args.data)
    else:
        optimize(args.optimize, args.data, args.definition)


def optimize(ntries, data, defn):

    for _ in range(ntries):
        launch(data, defn)
        time.sleep(0.5)

        
def launch(data, defn):

    script = job_script(data, defn)

    cmd = [
        'qsub',
        '-d', '/lcg/storage15/atlas/gagnon/work',
        '-N', 'optimization',
        '-joe',
    ]

    qsub = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    qsub.communicate(job_script(data, defn))

           
def train(defpath, datapath):    
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


def job_script(datapath, defpath):
    return """
mkdir ${PBS_JOBID}_${PBS_JOBNAME}
cd ${PBS_JOBID}_${PBS_JOBNAME}

git clone ~/dev/deep_susy/git .
. scripts/setup.sh

python2 scripts/launch.py --train --data %s --def %s > launch.log
""" % \
    (os.path.abspath(datapath), os.path.abspath(defpath))



 
     

if __name__ == '__main__':
    utils.main(main, 'launch')
