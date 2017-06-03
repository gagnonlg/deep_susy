get_dsid () {
    echo $1 | awk -F. '{print $3}'
}

DATADIR=/lcg/storage15/atlas/gagnon/data/multib_2.4.28
CODEDIR=/lcg/storage15/atlas/gagnon/work/2017-06-03_NNinput/deep_susy
OUTPUTDIR=/lcg/storage15/atlas/gagnon/work/2017-06-03_NNinput

for ddir in $(find $DATADIR -type d -name '*.root'); do
    dsid=$(get_dsid $(basename $ddir))
    cat <<EOF | qsub -d $OUTPUTDIR -N $dsid -joe
mkdir \${PBS_JOBID}_\${PBS_JOBNAME}
cd \${PBS_JOBID}_\${PBS_JOBNAME}
source $CODEDIR/scripts/setup.sh
python2 $CODEDIR/scripts/select_events.py \
	--dsid $dsid \
	--nsmall 10 --nlarge 4 --nlepton 4 \
	--inputs $ddir/*.root* \
	--data-version '2-4-28-0'
EOF
    sleep 0.5s
done
