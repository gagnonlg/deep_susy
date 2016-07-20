virtualenv --python=$(which python) venv_deepsusy
. venv_deepsusy/bin/activate

export CC=$(which gcc)
export CXX=$(which g++)

pip install numpy
pip install scipy
pip install h5py
pip install Theano
pip install keras

which root >/dev/null
if [ $? -eq 0 ]
then
    pip install root_numpy
    pip install rootpy
fi
