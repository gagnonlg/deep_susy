set -u

DATADIR=$1
CODEDIR=/lcg/storage15/atlas/gagnon/work/2017-06-19_deep-SUSY/deep_susy

cat <<EOF | qsub -d $PWD -N $(basename $DATADIR) -joe
mkdir \${PBS_JOBID}_\${PBS_JOBNAME}
cd \${PBS_JOBID}_\${PBS_JOBNAME}
source $CODEDIR/scripts/setup.sh
python2 $CODEDIR/scripts/current_dataset/create_h5.py $DATADIR
EOF
