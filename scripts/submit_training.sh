set -u

MODEL=$1
DATA=$2

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cat <<EOF | qsub -d $PWD -N $(basename $MODEL .py) -joe
source $DIR/setup.sh
pushd $DIR && git describe && popd
$DIR/theano_wrapper python2 $DIR/train_on_NNinput.py $MODEL $DATA
EOF
