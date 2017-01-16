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

if __name__ == '__main__':
    unittest.main()
