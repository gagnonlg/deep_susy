. /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
lsetup 'ROOT 6.06.02-x86_64-slc6-gcc49-opt'

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$(dirname $DIR)/submodules:$PYTHONPATH
