for name in $(cat retry.list); do
    cat <<'EOF' | qsub -N metrics -d /home/zp/gagnon/dev/deep_susy/git/baz -v name=$name
    . ../scripts/setup.sh
    dir=/lcg/storage15/atlas/gagnon/work/${name}_optimization
    rm -r $name
    mkdir $name
    cd $name
    cp $dir/GttNN_2.definition.txt .
    cp $dir/GttNN_2.keras_model.h5 .
    test -e $dir/GttNN_2.normalization.txt && cp $dir/GttNN_2.normalization.txt .
    python2 /home/zp/gagnon/dev/deep_susy/git/scripts/recompute_metrics.py \
	    GttNN_2.{definition.txt,keras_model.h5,normalization.txt} \
	    /lcg/storage15/atlas/gagnon/data/NNinput-a97618a-2.h5
    
EOF
    sleep 0.5s
done


