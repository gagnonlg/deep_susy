import argparse
import collections
import itertools
import os
import logging
import re
import tempfile
import unittest

import h5py as h5
import numpy as np
import root_numpy

from deep_susy import dataset, gtt, utils

LOG = logging.getLogger('test_dataset')
STATUS = 0

def Test_create_master(ddict, masterh5):
    class _Test_create_master(unittest.TestCase):

        def test_structure(self):
            self.assertTrue('signal' in masterh5.keys())
            self.assertTrue('background' in masterh5.keys())

            for key in ddict:
                grp = 'signal' if key.startswith('Gtt_') else 'background'
                subh5 = masterh5[grp]
                if key == 'Diboson':
                    self.assertFalse(key in subh5.keys())
                else:
                    self.assertTrue(key in subh5.keys(), '{}, {}')
                    self.assertTrue('input' in subh5[key].keys())
                    self.assertTrue('target' in subh5[key].keys())
                    self.assertTrue('metadata' in subh5[key].keys())

        def test_nevents(self):
            skeys = ['signal/' + k for k in masterh5['signal'].keys()]
            bkeys = ['background/' + k for k in masterh5['background'].keys()]
            for key in itertools.chain(skeys, bkeys):
                dkey = key.split('/')[-1]
                n = sum([d.tree.GetEntries() for d in ddict[dkey]])
                self.assertEqual(masterh5[key + '/input'].shape[0], n)
                self.assertEqual(masterh5[key + '/target'].shape[0], n)
                self.assertEqual(masterh5[key + '/metadata'].shape[0], n)

        def test_gtt_masses(self):

            skeys = ['signal/' + k + '/input' for k in masterh5['signal'].keys()]
            bkeys = ['background/' + k + '/input' for k in masterh5['background'].keys()]
            for key in itertools.chain(skeys, bkeys):
                dkey = key.split('/')[1]

                if key.startswith('signal'):
                    dsid = ddict[dkey][0].dsid
                    m_g, m_l = gtt.get_masses(int(dsid))
                else:
                    m_g = m_l = 0

                    self.assertTrue(
                        np.allclose(masterh5[key].value['I_m_gluino'], m_g),
                    )
                    self.assertTrue(
                        np.allclose(masterh5[key].value['I_m_lsp'], m_l),
                    )

        def long_test_branches_input_signal(self):
            self.branches_test("signal/Gtt_2000_5000_600/input")
        def long_test_branches_input_background(self):
            self.branches_test("background/ttbar/input")
        def long_test_branches_metadata_signal(self):
            self.branches_test("signal/Gtt_2000_5000_600/metadata")
        def long_test_branches_metadata_background(self):
            self.branches_test("background/ttbar/metadata")

        def branches_test(self, key):
            dkey = key.split('/')[1]
            indices = utils.range_sequence(
                [d.tree.GetEntries() for d in ddict[dkey]]
            )
            for d, (i0, i1) in zip(ddict[dkey], indices):
                array = root_numpy.tree2array(d.tree)
                prefix = 'I_' if key.endswith('input') else 'M_'
                branches = [b for b,_ in array.dtype.descr if b.startswith(prefix)]

                if prefix == 'I_':
                    branches.remove('I_m_gluino')
                    branches.remove('I_m_lsp')

                for br in branches:
                    self.assertTrue(
                        np.allclose(array[br], masterh5[key][i0:i1][br]),
                        'key:{}, br:{}'.format(key, br)
                    )

        def test_targets(self):
            skeys = ['signal/' + k + '/target' for k in masterh5['signal'].keys()]
            bkeys = ['background/' + k + '/target' for k in masterh5['background'].keys()]
            for key in itertools.chain(skeys, bkeys):
                tgt = masterh5[key][0]
                self.assertTrue(np.allclose(np.tile(tgt, reps=(masterh5[key].shape[0],1)), masterh5[key]))
                self.assertTrue(tgt.shape[0] > 1)
                i1 = np.where(tgt == 1)[0]
                self.assertEqual(i1.shape[0], 1)
                i1 = i1[0]
                if key.startswith('signal'):
                    self.assertEqual(i1, 0)
                    self.assertTrue(np.allclose(tgt[1:], 0))
                else:
                    self.assertNotEqual(i1, 0)
                    self.assertTrue(np.allclose(tgt[:i1], 0))
                    self.assertTrue(np.allclose(tgt[(i1+1):], 0))


    return _Test_create_master


def Test_create_split(masterh5, splith5):

    class _Test_create_split(unittest.TestCase):
        def setUp(self):
            self.skeys = ['signal/' + k for k in masterh5['signal'].keys()]
            self.bkeys = ['background/' + k for k in masterh5['background'].keys()]

        def test_structure(self):
            for fold in ['training', 'validation', 'test']:
                self.assertTrue(fold in splith5.keys())
                dset = splith5[fold]
                for tp in ['signal', 'background']:
                    self.assertTrue(tp in dset.keys())
                    for key in dset[tp].keys():
                        full = '/'.join([fold, tp, key])
                        self.assertTrue('input' in splith5[full])
                        self.assertTrue('target' in splith5[full])
                        self.assertTrue('metadata' in splith5[full])

        def test_nevents(self):
            for key in itertools.chain(self.skeys, self.bkeys):
                non_split = masterh5[key + '/input'].shape[0]
                split = 0;
                for fold in ['training', 'validation', 'test']:
                    split += splith5[fold + '/' + key + '/input'].shape[0]
                self.assertEqual(split, non_split)

        def test_weights(self):
            self.skeys = ['signal/' + k for k in masterh5['signal'].keys()]
            self.bkeys = ['background/' + k for k in masterh5['background'].keys()]
            for key in itertools.chain(self.skeys, self.bkeys):
                non_split = np.sum(masterh5[key + '/metadata']['M_weight'])
                for fold in ['training', 'validation', 'test']:
                    mdset = splith5[fold + '/' + key + '/metadata']
                    if mdset.size > 0:
                        split = np.sum(mdset['M_weight'])
                        self.assertTrue(
                            np.isclose(split, non_split),
                            'key={}, fold={}, split={}, non-split={}'.format(key, fold, split, non_split)
                        )

        def long_test_masses(self):
            mdict = collections.defaultdict(float)
            wtot = 0
            for fold in ['training', 'validation', 'test']:
                for key in self.skeys:
                    mg = splith5[fold + '/' + key + '/input'][0]['I_m_gluino']
                    ml = splith5[fold + '/' + key + '/input'][0]['I_m_lsp']
                    w =  splith5[fold + '/' + key + '/metadata'][0]['M_weight']
                    mdict[(mg,ml)] = w
                    wtot += w
                mdict = {k: v / wtot for (k,v) in mdict.iteritems()}

                b_mdict = collections.defaultdict(float)
                b_wtot = 0
                for key in self.bkeys:
                    ids = splith5[fold + '/' + key + '/input']
                    mds = splith5[fold + '/' + key + '/metadata']
                    if ids.shape[0] > 0:
                        masses = np.unique(ids.value[['I_m_gluino', 'I_m_lsp']])
                        for i in range(masses.shape[0]):
                            mg = masses[i]['I_m_gluino']
                            ml = masses[i]['I_m_lsp']
                            isel = np.where((ids['I_m_gluino'] == mg)&(ids['I_m_lsp'] == ml))[0]
                            w = np.sum(mds['M_weight'][isel])
                            b_mdict[(mg,ml)] += w
                            b_wtot += w
                b_mdict = {k: v / b_wtot for (k,v) in b_mdict.iteritems()}

                for masses in mdict:
                    self.assertTrue(
                        np.isclose(mdict[masses], b_mdict[masses], rtol=15.0),
                        '{}/{}: {}, {}'.format(fold, masses, mdict[masses], b_mdict[masses])
                    )

        def long_test_duplicated_events(self):
            for key in itertools.chain(self.skeys, self.bkeys):
                runs = {}
                evts = {}
                for fold in ['training', 'validation', 'test']:
                    mds = splith5[fold + '/' + key + '/metadata']
                    if mds.size > 0:
                        runs[fold] = mds['M_run_number']
                        evts[fold] = mds['M_event_number']

                def _check(f1, f2):
                    if f1 in runs and f2 in runs:
                        for run in np.unique(runs[f1]):
                            evt1 = evts[f1][np.where(runs[f1] == run)]
                            evt2 = evts[f2][np.where(runs[f2] == run)]
                            inter = np.intersect1d(evt1, evt2)

                            if not inter.shape[0] == 0:
                                # Checking kinematics'

                                if1 = splith5[f1 + '/' + key + '/input'].value
                                if2 = splith5[f2 + '/' + key + '/input'].value

                                dupi = None
                                for dup in inter:
                                    d1 = if1[np.where((runs[f1] == run)&(evts[f1] == dup))]
                                    d2 = if2[np.where((runs[f2] == run)&(evts[f2] == dup))]
                                    if dupi is None:
                                        dupi = np.array([], dtype=d1.dtype)
                                    dupi = np.concatenate([dupi, d1])
                                    dupi = np.concatenate([dupi, d2])

                                dupi_u = np.unique(dupi)

                                self.assertTrue(
                                    dupi_u.shape == dupi.shape,
                                    "duplicated events with same kinematics: key={}, f1={}, f2={}".format(key, f1, f2)
                                )

                                # different kinematics: OK

                _check('training', 'validation')
                _check('training', 'test')
                _check('validation', 'test')


    return _Test_create_split

def Test_unpack(splith5):

    def _get_h5_keys(h5dset, fold, dsetname):
        keys = []
        def _select(key):
            if fold in key and dsetname in key:
                keys.append(key)
        h5dset.visit(_select)
        return keys


    class _Test_unpack(unittest.TestCase):

        def _test_structured(self, dname):
            for fold in ['training', 'validation', 'test']:
                x = dataset.unpack(splith5, fold, dname)
                i_0 = i_1 = 0
                for key in _get_h5_keys(splith5, fold, dname):
                    xinit = splith5[key]
                    i_1 += xinit.shape[0]
                    if i_1 > i_0:
                        res = []
                        for i, (name, _) in enumerate(xinit.dtype.descr):
                            self.assertTrue(np.allclose(x[i_0:i_1, i], xinit[name]))
                    i_0 = i_1

        def long_test_input(self):
            self._test_structured('input')
        def test_metadata(self):
            self._test_structured('metadata')

        def test_target(self):
            for fold in ['training', 'validation', 'test']:
                x = dataset.unpack(splith5, fold, 'target')
                i_0 = i_1 = 0
                for key in _get_h5_keys(splith5, fold, 'target'):
                    xinit = splith5[key]
                    i_1 += xinit.shape[0]
                    if i_1 > i_0:
                        self.assertTrue(np.allclose(x[i_0:i_1], xinit))
                    i_0 = i_1

    return _Test_unpack

class Test_config(unittest.TestCase):
    def test_config(self):
        path = utils.project_path('datasets.config')
        with open(path, 'r') as cfg:
            lines = cfg.readlines()
            seen = []
            i = 0
            while i < len(lines):
                if lines[i].startswith('#'):
                    self.assertFalse(lines[i] in seen, lines[i])
                    seen.append(lines[i])
                m1 = re.match('^# Gtt_(.*)_5000_(.*)$', lines[i])
                if m1 is not None:
                    i += 1
                    m2 = re.search('ttn1_(.*)_5000_(.*)$', lines[i])
                    self.assertTrue(m2 is not None)
                    self.assertEqual(m1.group(1), m2.group(1))
                    self.assertEqual(m1.group(2), m2.group(2))
                i += 1

def _add_tests(suite, test_cls, short_only):
    for mthd in dir(test_cls):
        if mthd.startswith('test_'):
            suite.addTest(test_cls(mthd))
        if not short_only and mthd.startswith('long_test_'):
            suite.addTest(test_cls(mthd))


def _run_tests(datadir, masterh5, splith5, short_only, fail_fast, skip_master, skip_split):
    tests = unittest.TestSuite()
    if not skip_master:
        _add_tests(tests, Test_create_master(datadir, masterh5), short_only)
    if not skip_split:
        _add_tests(tests, Test_create_split(masterh5, splith5), short_only)
        _add_tests(tests, Test_unpack(splith5), short_only)
    _add_tests(tests, Test_config, short_only)
    unittest.TextTestRunner(failfast=fail_fast, verbosity=2).run(tests)


def _main():
    args = argparse.ArgumentParser()
    args.add_argument('datadir')
    args.add_argument('--master')
    args.add_argument('--split')
    args.add_argument('--short-only', action='store_true')
    args.add_argument('--fail-fast', action='store_true')
    args.add_argument('--skip-master', action='store_true')
    args.add_argument('--skip-split', action='store_true')
    args = args.parse_args()

    with tempfile.NamedTemporaryFile() as f_masterh5, \
         tempfile.NamedTemporaryFile() as f_splith5:

        if args.master is None:
            p_masterh5 = dataset.create_master(args.datadir, f_masterh5.name)
        elif not os.path.exists(args.master):
            p_masterh5 = dataset.create_master(args.datadir, args.master)
        else:
            p_masterh5 = args.master

        if args.split is None:
            p_splith5 = dataset.create_split(
                p_masterh5,
                f_splith5.name,
                custom_fractions={
                    # 'Diboson': (0, 0, 0),
                    'PhHppEG_ttbar': (1, 0, 0),
                    'MGPy8EG_ttbar': (0, 1, 0),
                    'ttbar': (0, 0, 1)
                }
            )
        elif not os.path.exists(args.split):
            p_splith5 = dataset.create_split(
                p_masterh5,
                args.split,
                custom_fractions={
                    # 'Diboson': (0, 0, 0),
                    'PhHppEG_ttbar': (1, 0, 0),
                    'MGPy8EG_ttbar': (0, 1, 0),
                    'ttbar': (0, 0, 1)
                }
            )
        else:
            p_splith5 = args.split

        masterh5 = h5.File(p_masterh5, 'r')
        splith5 = h5.File(p_splith5, 'r')
        _run_tests(dataset.lookup(args.datadir, 'NNinput'), masterh5, splith5, args.short_only, args.fail_fast, args.skip_master, args.skip_split)

    return STATUS


if __name__ == '__main__':
    utils.main(_main, 'test_dataset.py')
