import collections
import itertools
import os

import h5py as h5
import matplotlib.pyplot as plt
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

def main():

    inputp = 'test.h5.4'
    outputp = 'test_split.h5'

    if not os.path.exists(outputp):
        outputp = dataset.create_split(
            inputp,
            outputp,
            custom_fractions={
                'PhHppEG_ttbar': (1, 0, 0),
                'MGPy8EG_ttbar': (0, 1, 0),
                'ttbar': (0, 0, 1)
            }
        )

    input_f = h5.File(inputp, 'r')
    output_f = h5.File(outputp, 'r')

    # test split number of events
    print '=> Number of events test'
    skeys = ['signal/' + k for k in input_f['signal'].keys()]
    bkeys = ['background/' + k for k in input_f['background'].keys()]
    for key in itertools.chain(skeys, bkeys):
        non_split = input_f[key + '/input'].shape[0]
        split = 0;
        for fold in ['training', 'validation', 'test']:
            split += output_f[fold + '/' + key + '/input'].shape[0]
        result = _result(split == non_split)
        print '  -> {}: {}, {}'.format(key, non_split, split)
        print '         ' + result

    # test weights
    print '=> Weights test'
    for key in itertools.chain(skeys, bkeys):
        non_split = np.sum(input_f[key + '/metadata']['M_weight'])
        for fold in ['training', 'validation', 'test']:
            mdset = output_f[fold + '/' + key + '/metadata']
            if mdset.size > 0:
                split = np.sum(mdset['M_weight'])
                result = _result(np.isclose(split, non_split))
                print '  -> {}: {}, {}'.format(fold + '/' + key, non_split, split, result)
                print '         ' + result

    # masses test
    print '=> Masses test'
    mdict = collections.defaultdict(float)
    wtot = 0
    for fold in ['training', 'validation', 'test']:
        for key in skeys:
            mg = output_f[fold + '/' + key + '/input'][0]['I_m_gluino']
            ml = output_f[fold + '/' + key + '/input'][0]['I_m_lsp']
            w =  output_f[fold + '/' + key + '/metadata'][0]['M_weight']
            if mg == 0 or ml == 0:
                print '****** null mass for ' + fold + '/' + key
            mdict[(mg,ml)] = w
            wtot += w
        mdict = {k: v / wtot for (k,v) in mdict.iteritems()}

        b_mdict = collections.defaultdict(float)
        b_wtot = 0
        for key in bkeys:
            ids = output_f[fold + '/' + key + '/input']
            mds = output_f[fold + '/' + key + '/metadata']
            if ids.shape[0] > 0:
                masses = np.unique(ids.value[['I_m_gluino', 'I_m_lsp']])
                for i in range(masses.shape[0]):
                    mg = masses[i]['I_m_gluino']
                    ml = masses[i]['I_m_lsp']
                    if mg == 0.0 or ml == 0.0:
                        print '****** null mass for ' + fold + '/' + key
                    isel = np.where((ids['I_m_gluino'] == mg)&(ids['I_m_lsp'] == ml))[0]
                    w = np.sum(mds['M_weight'][isel])
                    b_mdict[(mg,ml)] += w
                    b_wtot += w
        b_mdict = {k: v / b_wtot for (k,v) in b_mdict.iteritems()}

        for masses in mdict:
            if masses not in b_mdict and masses[0] != 0 and masses[1] != 0:
                continue
            result = _result(np.isclose(mdict[masses], b_mdict[masses], rtol=1.0)) # within a factor 2
            print '  -> {}/{}: {}, {}'.format(fold, masses, mdict[masses], b_mdict[masses])
            print '       ' + result

    print '==> duplicated event test'
    for key in itertools.chain(skeys, bkeys):
        runs = {}
        evts = {}
        for fold in ['training', 'validation', 'test']:
            mds = output_f[fold + '/' + key + '/metadata']
            if mds.size > 0:
                runs[fold] = mds['M_run_number']
                evts[fold] = mds['M_event_number']

        try:
            # train/valid
            isel = np.where(runs['training'] == runs['validation'])[0]
            result1 = not np.any(evts['training'][isel] == evts['validation'][isel])
            if not result1:
                if np.all(evts['training'][isel][np.where(evts['training'][isel] == evts['validation'][isel])] == 0):
                    print '**** WARNING: null event numbers in training/valid'
                    result1 = True
        except KeyError:
            result1 = True

        try:
            # train/test
            isel = np.where(runs['training'] == runs['test'])[0]
            result2 = not np.any(evts['training'][isel] == evts['test'][isel])
            if not result2:
                if np.all(evts['training'][isel][np.where(evts['training'][isel] == evts['test'][isel])] == 0):
                    print '**** WARNING: null event numbers in training/test'
                    result2 = True
        except KeyError:
            result2 = True

        try:
            # valid/test
            isel = np.where(runs['validation'] == runs['test'])[0]
            result3 = not np.any(evts['validation'][isel] == evts['test'][isel])
            if not result3:
                if np.all(evts['validation'][isel][np.where(evts['validation'][isel] == evts['test'][isel])] == 0):
                    print '**** WARNING: null event numbers in valid/test'
                    result3 = True
        except KeyError:
            result3 = True

        result = _result(result1 and result2 and result3)
        print '  -> {}:'.format(key)
        print '        ' + result


    if RESULT == 1:
        print '*** SOME TEST FAILED :( ***'
    else:
        print '*** ALL TEST SUCCEEDED :) ***'
    return RESULT


utils.main(main, 'test_split')
