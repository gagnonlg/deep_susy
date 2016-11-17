""" module for dataset-related code """

import logging

from dataset.merge import merge
from dataset.select import select, select_main
from dataset.split_tree import split, parallel_split

# This avoids missing modules (e.g. h5py) import errors while running
# only the select code in environment missing some packages (e.g. the
# grid)
try:
    from dataset.create import create, create_main
    from dataset.downsample import downsample, downsample_main
except ImportError:
    logging.basicConfig(format='[%(name)s] %(levelname)s %(message)s')
    logger = logging.getLogger('dataset.__init__.py')
    logger.warning('Unable to import from dataset.create')
