""" Some tests for the create_dataset code """
from array import array
import os
import shutil
import subprocess
import tempfile
import unittest

import h5py as h5
import numpy as np
import root_numpy
import ROOT

import create_dataset
import utils

class Test_split(unittest.TestCase):
    """ Some unit tests for the split() function """

    def setUp(self):
        self.__files = []

    def tearDown(self):
        for path in self.__files:
            os.remove(path)

    def __make_tfile(self, nevents):
        path = utils.uuid()[:-1] + '.root'
        tfile = ROOT.TFile(path, 'CREATE')
        tree = ROOT.TTree("dummy", "")
        tree_data = array('f', [0.0])
        tree.Branch('tree_data', tree_data, 'tree_data/F')
        for _ in range(nevents):
            tree.Fill()
        tree.Write()
        tfile.Close()

        self.__files.append(path)

        return path

    def test_missing_tree(self):
        """ test for invalid tree name detection """

        path = self.__make_tfile(1)
        self.assertRaises(
            IOError,
            create_dataset.split, path, 'dummyFOO', [1], ['bar']
        )

    def test_invalid_file(self):
        """ test for invalid tfile detection """

        self.assertRaises(
            IOError,
            create_dataset.split, utils.uuid()[-1], 'dummy', [1], ['bar']
        )

    def test_invalid_lengths(self):
        """ test for invalid tfile detection """

        self.assertRaises(
            ValueError,
            create_dataset.split, self.__make_tfile(1), 'dummy', [1,2], ['bar']
        )
  
       
    def __count(self, path):
        tfile = ROOT.TFile(path)
        tree = tfile.Get('dummy')
        n = tree.GetEntries()
        tfile.Close()
        return n

    def test_id(self):
        """ identity function splitting """
        path = self.__make_tfile(100)
        self.__files.append(path + '.1')
        create_dataset.split(path, 'dummy', [1], [path + '.1'])
        self.assertEqual(100, self.__count(path + '.1'))

    def test_split(self):
        """ 3-way splitting """
        path = self.__make_tfile(100)
        self.__files.append(path + '.1')
        self.__files.append(path + '.2')
        self.__files.append(path + '.3')
        create_dataset.split(path, 'dummy', [0.2,0.7,0.1], [path + '.%d'%i for i in [1,2,3]])
        self.assertEqual(0.2*100, self.__count(path + '.1'))
        self.assertEqual(0.7*100, self.__count(path + '.2'))
        self.assertEqual(0.1*100, self.__count(path + '.3'))

    def test_problematic_nevent(self):
        path = self.__make_tfile(347)
        self.__files.append(path+'.1')
        self.__files.append(path+'.2')
        create_dataset.split(path, 'dummy', [0.7,0.3], [path+'.1',path+'.2'])
        self.assertEqual(347, self.__count(path+'.1') + self.__count(path+'.2'))
        

    def test_renormalize(self):
        """ 3-way splitting with renormalization of fractions"""
        path = self.__make_tfile(100)
        self.__files.append(path + '.1')
        self.__files.append(path + '.2')
        self.__files.append(path + '.3')
        create_dataset.split(path, 'dummy', [0.4,1.4,0.2], [path + '.%d'%i for i in [1,2,3]])
        self.assertEqual(0.2*100, self.__count(path + '.1'))
        self.assertEqual(0.7*100, self.__count(path + '.2'))
        self.assertEqual(0.1*100, self.__count(path + '.3'))


class Test_root_to_h5(unittest.TestCase):
    """ Some tests for the root_to_h5() function """

    def setUp(self):
        self.__files = []

    def tearDown(self):
        for path in self.__files:
            os.remove(path)

    def test_invalid_file(self):

        self.assertRaises(
            IOError,
            create_dataset.root_to_h5,
            ''
        )

        self.assertRaises(
            IOError,
            create_dataset.root_to_h5,
            utils.uuid()[:-1]
        )

    def test_conversion(self):

        path = utils.uuid()[:-1] + '.root'
        self.__files.append(path)
        
        tfile = ROOT.TFile(path, 'CREATE')
        tree = ROOT.TTree("dummy", "")
        tree_data = array('f', [0.0])
        tree_data2 = array('f', [0.0])
        tree.Branch('tree_data', tree_data, 'tree_data[1]/F')
        tree.Branch('tree_data2', tree_data2, 'tree_data2[1]/F')
        for i in range(100):
            tree_data[0] = i
            tree_data2[0] = -1*i
            tree.Fill()
            
        tree.Write()
        tfile.Close()
   
        h5_path = create_dataset.root_to_h5(path)
        self.__files.append(h5_path)
        
        self.assertEqual(h5_path, os.path.abspath(path.replace('.root', '.h5')))

        h5file = h5.File(h5_path)
        dset = h5file['NNinput']

        self.assertEqual(dset.shape[0], 100)

        for i in range(100):
            self.assertEqual(dset['tree_data'][i], i)
            self.assertEqual(dset['tree_data2'][i], -i)


class Test_reweight(unittest.TestCase):

    def setUp(self):
        self.testfile = utils.top_directory() + '/test_data/370160.NNinput.root'
        create_dataset.split(
            self.testfile,
            'NNinput',
            [0.7, 0.3],
            [self.testfile +'.1', self.testfile+'.2']
        )
        self.path1 = create_dataset.root_to_h5(self.testfile+'.1')
        self.path2 = create_dataset.root_to_h5(self.testfile+'.2')

    def tearDown(self):
        os.remove(self.testfile+'.1')
        os.remove(self.testfile+'.2')
        os.remove(self.path1)
        os.remove(self.path2)

        
    def test_reweight(self):

        warray = root_numpy.root2array(self.testfile, branches='M_weight')
        xsec = np.sum(warray)

        h5_1 = h5.File(self.path1)
        h5_2 = h5.File(self.path2)

        ds_1 = h5_1['NNinput']['M_weight']
        ds_2 = h5_2['NNinput']['M_weight']

        wsum = np.sum(ds_1) + np.sum(ds_2)

        self.assertTrue(
            np.isclose(wsum,  xsec),
            'wsum:{}, xsec:{}'.format(wsum,xsec)
        )

        h5_1.close()
        h5_2.close()
                         
        create_dataset.reweight([self.path1, self.path2])

        h5_1 = h5.File(self.path1)
        h5_2 = h5.File(self.path2)

        ds_1 = h5_1['NNinput']['M_weight']
        ds_2 = h5_2['NNinput']['M_weight']

        wsum1 = np.sum(ds_1)
        wsum2 = np.sum(ds_2)
        
        self.assertTrue(
            np.isclose(wsum1, xsec),
            'wsum1:{}, xsec:{}'.format(wsum1,xsec)
        )
        self.assertTrue(
            np.isclose(wsum2, xsec),
            'wsum2:{}, xsec:{}'.format(wsum2,xsec)
        )
        

        h5_1.close()
        h5_2.close()

class Test_prepare_for_merge(unittest.TestCase):

    def setUp(self):
        self.testfile = utils.top_directory() + '/test_data/370160.NNinput.root'
        self.__files = []

    def tearDown(self):
        for f in self.__files:
            os.remove(f)
        
    def test_invalid_args(self):

        self.assertRaises(
            IOError,
            create_dataset.prepare_for_merge,
            utils.uuid()[:-1], [0.5, 0.25, 0.25]
        )

        self.assertRaises(
            ValueError,
            create_dataset.prepare_for_merge,
            self.testfile, [1.0]
        )

    def test_conversion(self):

        paths = create_dataset.prepare_for_merge(self.testfile, [0.5, 0.25, 0.25])
        self.__files += paths

        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(
                'Hierarchical Data Format (version 5) data'
                in subprocess.check_output(['file', p])
            )

        self.assertTrue(paths[0].endswith('.training.h5'))
        self.assertTrue(paths[1].endswith('.validation.h5'))
        self.assertTrue(paths[2].endswith('.test.h5'))

        # get the total xsec for the input
        tf = ROOT.TFile(self.testfile, 'READ')
        tr = tf.Get('NNinput')
        tr.Draw('M_weight>>hweight',)
        nevents = tr.GetEntries()
        hweight = ROOT.gDirectory.Get('hweight')
        xsec = hweight.GetMean() * hweight.GetEntries()

        # verify the match
        f0 = h5.File(paths[0])
        f1 = h5.File(paths[1])
        f2 = h5.File(paths[2])

        d0 = f0['NNinput']
        d1 = f1['NNinput']
        d2 = f2['NNinput']

        s0 = d0.shape[0]
        s1 = d1.shape[0]
        s2 = d2.shape[0]

        self.assertEqual(nevents, s0 + s1 + s2)
        self.assertEqual(round(0.5*nevents), s0)
        self.assertEqual(round(0.25*nevents), s1)
        self.assertTrue(
            round(0.25*nevents) == s2 or
            nevents - s0 - s1 == s2
        )

        w0 = np.sum(d0['M_weight'])
        w1 = np.sum(d0['M_weight'])
        w2 = np.sum(d0['M_weight'])

        self.assertTrue(np.isclose(w0, xsec))
        self.assertTrue(np.isclose(w1, xsec))
        self.assertTrue(np.isclose(w2, xsec))
        
        
        

if __name__ == '__main__':
    ROOT.gROOT.SetBatch()
    unittest.main()
