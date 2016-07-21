# setup and activate the virtualenv, making sure that the first python
# on PATH is picked-up
virtualenv --python=$(which python2) venv_deepsusy
. venv_deepsusy/bin/activate

# in case any of the package have to build from source, make sure that
# the correct gcc version is picked-up
export CC=$(which gcc)
export CXX=$(which g++)

# make sure that pip is up-to-date. if not, it will not download
# wheels and installation time will be very long
pip install --upgrade pip

# installing keras will pick up the required dependencies such as
# numpy/scipy, theano and pyyaml
pip install keras

# h5py is an optional dependency
pip install h5py

# install ROOT-specific packages if ROOT is installed
which root >/dev/null
if [ $? -eq 0 ]
then
    pip install root_numpy
    pip install rootpy
fi
