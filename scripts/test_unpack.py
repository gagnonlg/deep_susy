from __future__ import print_function

import itertools

import h5py as h5
import numpy as np

import dataset
import utils

RESULT = 0
def _result(p):
    global RESULT
    if p:
        return 'SUCCESS'
    else:
        RESULT = 1
        return '***FAILURE***'


def _get_h5_keys(h5dset, fold, dsetname):
    keys = []

    def _select(key):
        if fold in key and dsetname in key:
            keys.append(key)
    h5dset.visit(_select)
    return keys

def main():

    inputp = 'test_split.h5'
    inputf = h5.File(inputp)

    folds = ['training', 'validation', 'test']
    dnames = ['input', 'target', 'metadata']
    for fold, dname in itertools.product(folds, dnames):
        print('=> ' + fold + ', ' + dname)
        x = dataset.unpack(inputf, fold, dname)
        i_0 = i_1 = 0
        for key in _get_h5_keys(inputf, fold, dname):
            print('  -> ' + key + ': ', end='')
            xinit = inputf[key]
            i_1 += xinit.shape[0]
            if i_1 > i_0:
                if dname == 'target':
                    result = _result(np.allclose(x[i_0:i_1], xinit))
                else:
                    res = []
                    for i, (name, _) in enumerate(xinit.dtype.descr):
                        res.append(np.allclose(x[i_0:i_1, i], xinit[name]))
                    result = _result(all(res))
            else:
                result = 'SKIP'
            print(result)
            i_0 = i_1

    return RESULT

utils.main(main, 'test_unpack')
