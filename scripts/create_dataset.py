import os
import logging

import h5py as h5
import numpy as np
import ROOT
import root_numpy

LOGGER = logging.getLogger('create_dataset')


########################################################################
# TTree splitting

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

    # we must use '==' instead of 'is' here because of PyROOT
    if intree == None:  # noqa pylint: disable=singleton-comparison
        raise IOError(
            'Unable to get tree "{}" from file {}'.format(treename, inpath)
        )

    __split_tree_by_fractions(intree, fractions, names)


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
        nentries = int(round(fraction * ntot))
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


########################################################################
# root -> hdf5

def root_to_h5(path):
    """ convert root file to h5 format """

    array = root_numpy.root2array(path)

    # output path
    outpath = os.path.basename(path)
    if outpath.endswith('.root'):
        outpath = outpath.replace('.root', '.h5')
    else:
        outpath += '.h5'

    h5_file = h5.File(outpath, 'w')
    h5_file.create_dataset(
        'NNinput',
        data=array
    )

    h5_file.close()

    return outpath

########################################################################
# reweighting

def reweight(paths):

    h5files = [h5.File(p, 'r+') for p in paths]
    dsets = [h5f['NNinput']['M_weight'] for h5f in h5files]

    wsums = []
    for wset in dsets:
        wsums.append(np.sum(wset))

    wsumtot = np.sum(wsums)

    for h5f, wsum in zip(h5files, wsums):
        h5f['NNinput']['M_weight'] *= (float(wsumtot) / wsum)
        h5f.close()
    
