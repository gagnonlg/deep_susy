set -e
set -u

NAME=$1
JOBLIST=$2
DATA=$3
OUTPUT_DIR=$4

####################

mkdir $OUTPUT_DIR

for jobid in $(cat $JOBLIST); do
    cat <<'EOF' | qsub -N ${jobid}_metrics -d $OUTPUT_DIR -v jobid=$jobid,name=$NAME,data=$DATA
    . ../scripts/setup.sh
    dir=/lcg/storage15/atlas/gagnon/work/${jobid}_optimization
    mkdir $jobid 
    cd $jobid
    git clone /home/zp/gagnon/dev/deep_susy/git .
    scripts/theano_wrapper python2 scripts/recompute_metrics.py \
      $dir/$name.{definition.txt,keras_model.h5,normalization.txt} \
      $data
EOF
    sleep 0.5s
done

# example:
# bash batch_scripts/recompute_all.sh GttNN_4 optimizations/jobs_round_4.txt /lcg/storage15/atlas/gagnon/data/NNinput-a42cfc0.h5 $PWD/round4

