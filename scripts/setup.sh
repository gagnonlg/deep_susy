SETUP_ATLAS=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
if [ -f $SETUP_ATLAS ]
then
    . /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
    lsetup 'ROOT 6.04.18-x86_64-slc6-gcc49-opt'
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOPDIR=$(dirname $DIR)
export PYTHONPATH=$TOPDIR:$TOPDIR/submodules:$PYTHONPATH
export PATH=$TOPDIR/bin:$PATH
