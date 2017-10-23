export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
SETUP_ATLAS=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
if [ -f $SETUP_ATLAS ]
then
    . /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
    lsetup 'ROOT 6.08.06-x86_64-slc6-gcc62-opt'
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOPDIR=$(dirname $DIR)
export PYTHONPATH=$TOPDIR:$TOPDIR/submodules:$PYTHONPATH
export PATH=$TOPDIR/bin:$PATH
