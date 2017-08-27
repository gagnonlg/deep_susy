""" Test k-fold CV """
import unittest

import numpy as np

from cross_validation import __partition as partition
from cross_validation import *


class DummyModel(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return self

    def test(self, data):
        return 0


# class TestKFold(unittest.TestCase):
#     def test_k_fold(self):
#         data = (np.arange(10), np.arange(10))
#         _, should_be_zero = k_fold(data, DummyModel(), nfolds=10)
#         self.assertEqual(should_be_zero[0], 0)
#         self.assertEqual(should_be_zero[1], 0)

class TestPartition(unittest.TestCase):

    def lpartition(self, d, n):
        return list(partition(d, n))

    def test_no_folds(self):
        data = (np.array([]), np.array([]))
        self.assertRaises(
            ValueError,
            self.lpartition,
            data, 0
        )
        self.assertRaises(
            ValueError,
            self.lpartition,
            data, -10
        )

    def test_fold_too_big(self):

        data = np.arange(10)
        self.assertRaises(
            ValueError,
            self.lpartition,
            (data, data), 11
        )

    def test_bad_data(self):
        datax = np.arange(10)
        datay = np.arange(11)
        self.assertRaises(
            ValueError,
            self.lpartition,
            (datax, datay), 10
        )
        self.assertRaises(
            ValueError,
            self.lpartition,
            datax, 10
        )
        self.assertRaises(
            TypeError,
            self.lpartition,
            ('foo', 'bar'), 10
        )

    def test_fold_size(self):

        datax = np.arange(10)
        datay = np.arange(10)+100
        data = (datax, datay)

        self.assertEqual(len(list(partition(data, 10))), 10)

        for (x0, y0), (x1, y1) in partition(data, 10):
            self.assertEqual(x0.shape[0], 9)
            self.assertEqual(y0.shape[0], 9)
            self.assertEqual(x1.shape[0], 1)
            self.assertEqual(y1.shape[0], 1)

    def test_uneven_fold_size(self):
        datax = np.arange(5)
        data = (datax, datax)

        self.assertEqual(len(list(partition(data, 4))), 4)

    def test_fold_size_random(self):

        for _ in range(100):

            size = np.random.randint(10, 1000)
            datax = np.random.random(size)
            datay = np.random.random(size)
            data = (datax, datay)

            nfolds = np.random.randint(1, size)

            nfolds_returned = len(list(partition(data, nfolds)))
            self.assertEqual(
                nfolds_returned,
                nfolds,
                msg='size={}, nfolds={}, nfolds_returned={}'.format(
                    size,
                    nfolds,
                    nfolds_returned
                )
            )

            x1size = 0
            y1size = 0
            for _, (x1, y1) in partition(data, 10):
                x1size += x1.shape[0]
                y1size += y1.shape[0]

            self.assertEqual(x1size, size)
            self.assertEqual(y1size, size)


if __name__ == '__main__':
    unittest.main()
