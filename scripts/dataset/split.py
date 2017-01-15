""" code to split a TFile/TTree into many fractions """
import os
import logging
from multiprocessing.pool import ThreadPool

import ROOT

LOGGER = logging.getLogger('dataset.split')


def split(inpath, treename, fractions, names):
    """ split a tree into many fractions

    Args:
      inpath: path to ROOT file containing tree to split
      treename: name of the tree to split
      fractions: list of fractions to compute the split sizes
      names: list of output path. must be of same length than `fractions`
    Returns:
      None
    Raises:
      ValueError: Length of `fractions` does not match length of `names`
      IOError: Unable to open the ROOT file or get the tree
    """
    LOGGER.info('splitting tree "%s" from file %s', treename, inpath)

    infile = ROOT.TFile(inpath, 'READ')  # pylint: disable=no-member
    if infile.IsZombie():
        raise IOError('Unable to open file: {}'.format(inpath))

    intree = infile.Get(treename)
    if intree is None:
        raise IOError(
            'Unable to get tree "{}" from file {}'.format(treename, inpath)
        )

    __split_tree_by_fractions(intree, fractions, names)


def parallel_split(inpaths, treename, fractions, outdirs, nthreads=10):
    """ split many trees into many fractions in parallel

    Args:
      inpaths: list of paths to ROOT file containing tree to split
      treename: name of the tree to split
      fractions: list of fractions to compute the split sizes
      outdirs: list of output directories to store the splits
      nthreads: number of jobs to run in parallel
    Returns:
      list of `split` results
    Raises:
      ValueError: Length of `fractions` does not match length of `names`
      IOError: Unable to open the ROOT file or get the tree
    """
    def _split(path):
        names = ['{}/{}'.format(outd, os.path.basename(path))
                 for outd in outdirs]
        return split(path, treename, fractions, names)

    pool = ThreadPool(nthreads)
    results = pool.map(_split, inpaths)
    pool.close()
    pool.join()

    return results


def __split_tree_by_fractions(tree, fractions, names):
    # Check this explicitely, otherwise the call to zip() will
    # silently drop some fractions or names
    if len(fractions) != len(names):
        raise ValueError('len(fractions) != len(names)')

    if sum(fractions) != 1:
        LOGGER.warning('normalizing fractions to unity')
        norm = 1.0 / sum(fractions)
        fractions = [norm * f for f in fractions]

    ntot = tree.GetEntries()
    start = 0
    for name, fraction in zip(names, fractions):
        nentries = int(fraction * ntot)
        LOGGER.info(
            '%.2f%% (%d to %d = %d) into %s',
            fraction*100,
            start,
            start+nentries,
            nentries,
            name
        )
        __split_tree(tree, name, start, nentries)
        start += nentries


def __split_tree(tree, output, start, nentries):
    # pylint: disable=no-member,unused-variable

    tfile = ROOT.TFile(output, 'CREATE')
    if tfile.IsZombie():
        raise RuntimeError('FIXME')

    new_tree = tree.CopyTree("", "", nentries, start)

    tfile.Write("", ROOT.TObject.kWriteDelete)
