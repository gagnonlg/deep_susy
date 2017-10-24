import argparse
import logging
import os
import random
import subprocess

from deep_susy import utils


SCRIPT_TEMPLATE = """
cd $SCRATCH
mkdir ${{PBS_JOBID}}_${{PBS_JOBNAME}} && cd ${{PBS_JOBID}}_${{PBS_JOBNAME}}
module load CUDA
source {codedir}/scripts/setup.sh
export THEANO_FLAGS="floatX=float32,compiledir=$LSCRATCH,device=gpu"
export PARAMETRIZATION={parametrization}
python2 {codedir}/scripts/train_on_NNinput.py \
	--logfile ${{PBS_JOBID}}_${{PBS_JOBNAME}}_train.log \
	{codedir}/models/generators/gen_002.py \
        {dset}
python2 {codedir}/scripts/apply_model.py \
	--logfile ${{PBS_JOBID}}_${{PBS_JOBNAME}}_apply-training.log \
        *_trained.h5 \
        {dset} \
        training
python2 {codedir}/scripts/apply_model.py \
	--logfile ${{PBS_JOBID}}_${{PBS_JOBNAME}}_apply-validation.log \
        *_trained.h5 \
        {dset} \
        validation
"""


def make_script(dset):
    return SCRIPT_TEMPLATE.format(
        codedir=utils.top_directory(),
        parametrization=('ptetaphim' if 'ptetaphim' in dset else 'pxpypze'),
        dset=dset
    )


def submit_one(dset, i, dry_run):
    cmd = [
        'qsub',
        '-d', os.getcwd(),
        '-N', 'gen002_{}'.format(i),
        '-l', 'walltime=96:00:00,nodes=1:ppn=1,mem=4gb',
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    script = make_script(dset)
    if dry_run:
        path = 'optim_gen002_script_{}.sh'.format(i)
        with open(path, 'w') as sfile:
            sfile.write(script)
        logging.info('(dry-run mode) wrote script: %s', path)
    else:
        jobid, _ = proc.communicate(script)
        logging.info('PBS_JOBID=%s', jobid)


def submit_all(njobs, dsets, dry_run):
    for i in range(njobs):
        submit_one(random.choice(dsets), i, dry_run)


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dry-run', action='store_true')
    args.add_argument('NJOBS', type=int)
    args.add_argument('DATASETS', nargs='+')
    return args.parse_args()


def main():
    args = get_args()
    logging.info('NJOBS: %d', args.NJOBS)
    logging.info('DATASETS: %s', args.DATASETS)
    submit_all(args.NJOBS, args.DATASETS, args.dry_run)


if __name__ == '__main__':
    utils.main(main, '')
