""" module for dataset-related code """

from dataset.split_tree import split, parallel_split
import dataset.merge

__all__ = [
    'split',
    'parallel_split',
    'merge'
]
