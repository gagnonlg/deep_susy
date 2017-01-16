""" Some tests for the dataset.split module """
from array import array
import os
import unittest

import ROOT

import dataset
import utils


class TestSplit(unittest.TestCase):
    """ Some unit tests for the dataset.split() function """
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
            dataset.split, path, 'dummyFOO', [1], ['bar']
        )

    def test_invalid_file(self):
        """ test for invalid tfile detection """

        self.assertRaises(
            IOError,
            dataset.split, utils.uuid()[-1], 'dummy', [1], ['bar']
        )

    def test_invalid_lengths(self):
        """ test for invalid tfile detection """

        self.assertRaises(
            ValueError,
            dataset.split, self.__make_tfile(1), 'dummy', [1,2], ['bar']
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
        dataset.split(path, 'dummy', [1], [path + '.1'])
        self.assertEqual(100, self.__count(path + '.1'))

    def test_split(self):
        """ 3-way splitting """
        path = self.__make_tfile(100)
        self.__files.append(path + '.1')
        self.__files.append(path + '.2')
        self.__files.append(path + '.3')
        dataset.split(path, 'dummy', [0.2,0.7,0.1], [path + '.%d'%i for i in [1,2,3]])
        self.assertEqual(0.2*100, self.__count(path + '.1'))
        self.assertEqual(0.7*100, self.__count(path + '.2'))
        self.assertEqual(0.1*100, self.__count(path + '.3'))

    def test_renormalize(self):
        """ 3-way splitting with renormalization of fractions"""
        path = self.__make_tfile(100)
        self.__files.append(path + '.1')
        self.__files.append(path + '.2')
        self.__files.append(path + '.3')
        dataset.split(path, 'dummy', [0.4,1.4,0.2], [path + '.%d'%i for i in [1,2,3]])
        self.assertEqual(0.2*100, self.__count(path + '.1'))
        self.assertEqual(0.7*100, self.__count(path + '.2'))
        self.assertEqual(0.1*100, self.__count(path + '.3'))


if __name__ == '__main__':
    unittest.main()
