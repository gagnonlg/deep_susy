""" code to split a TFile/TTree into many fractions """
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

    if len(fractions) != len(names):
        raise ValueError('len(fractions) != len(names)')

    infile = ROOT.TFile(inpath, 'READ')  # pylint: disable=no-member
    if infile.IsZombie():
        raise IOError('Unable to open file: {}'.format(inpath))

    intree = infile.Get(treename)
    if intree == None:  # noqa pylint: disable=singleton-comparison
        raise IOError(
            'Unable to get tree "{}" from file {}'.format(treename, inpath)
        )

    split_tree_by_fractions_(intree, fractions, names)


def split_tree_by_fractions_(tree, fractions, names):
    """ split a tree into many fractions """

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
        split_tree_(tree, name, start, nentries)
        start += nentries


def split_tree_(tree, output, start, nentries):
    """ copy subset of tree """
    # pylint: disable=no-member,unused-variable

    tfile = ROOT.TFile(output, 'CREATE')
    if tfile.IsZombie():
        raise RuntimeError('FIXME')

    new_tree = tree.CopyTree("", "", nentries, start)

    tfile.Write("", ROOT.TObject.kWriteDelete)


def parallel_split(inpaths, treename, fractions, nametrans, nthreads=10):
    """ split many trees into many fractions in parallel

    Args:
      inpaths: list of paths to ROOT file containing tree to split
      treename: name of the tree to split
      fractions: list of fractions to compute the split sizes
      nametrans: list of (to_replace, replacement) pairs to create output names
      nthreads: number of jobs to run in parallel
    Returns:
      list of `split` results
    Raises:
      ValueError: Length of `fractions` does not match length of `names`
      IOError: Unable to open the ROOT file or get the tree
    """
    def _split(path):
        names = [path.replace(str0, str1) for (str0, str1) in nametrans]
        return split(path, treename, fractions, names)

    pool = ThreadPool(nthreads)
    results = pool.map(_split, inpaths)
    pool.close()
    pool.join()

    return results
