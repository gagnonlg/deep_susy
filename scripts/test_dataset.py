import argparse
import itertools
import os
import logging
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

        def atest_gtt_masses(self):

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
                branches = [b for b,_ in array.dtype.descr if b.startswith('M_')]

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

def _run_tests(datadir, masterh5, splith5, short_only=False):

    tests = unittest.TestSuite()

    test_master = Test_create_master(datadir, masterh5)
    for mthd in dir(test_master):
        if mthd.startswith('test_'):
            tests.addTest(test_master(mthd))
        if not short_only and mthd.startswith('long_test_'):
            tests.addTest(test_master(mthd))
    unittest.TextTestRunner(verbosity=2).run(tests)


def _main():
    args = argparse.ArgumentParser()
    args.add_argument('datadir')
    args.add_argument('--master')
    args.add_argument('--split')
    args.add_argument('--short-only', action='store_true')
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
                    'Diboson': (0, 0, 0),
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
                    'Diboson': (0, 0, 0),
                    'PhHppEG_ttbar': (1, 0, 0),
                    'MGPy8EG_ttbar': (0, 1, 0),
                    'ttbar': (0, 0, 1)
                }
            )
        else:
            p_splith5 = args.split

        masterh5 = h5.File(p_masterh5, 'r')
        splith5 = h5.File(p_splith5, 'r')
        _run_tests(dataset.lookup(args.datadir, 'NNinput'), masterh5, splith5, args.short_only)

    return STATUS


if __name__ == '__main__':
    utils.main(_main, 'test_dataset.py')
