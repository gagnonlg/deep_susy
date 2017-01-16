""" Some tests for the create_dataset code """
from array import array
import os
import shutil
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
        
        self.assertEqual(h5_path, path.replace('.root', '.h5'))

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
        

if __name__ == '__main__':
    ROOT.gROOT.SetBatch()
    unittest.main()
