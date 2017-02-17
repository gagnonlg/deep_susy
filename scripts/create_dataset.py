""" Create an HDF5 dataset from flat ROOT files """
import argparse
import logging
import os
import shutil
import tempfile

import h5py as h5
import numpy as np
import ROOT
import root_numpy

import utils

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

    return os.path.abspath(outpath)


########################################################################
# reweighting

def reweight(paths):
    """ Reweight the "M_weight" variable to sum between all files """

    h5files = [h5.File(p, 'r+') for p in paths]
    dsets = [h5f['NNinput']['M_weight'] for h5f in h5files]

    wsums = []
    for wset in dsets:
        wsums.append(np.sum(wset))

    wsumtot = np.sum(wsums)

    for h5f, wsum in zip(h5files, wsums):
        h5f['NNinput']['M_weight'] *= (float(wsumtot) / wsum)
        h5f.close()


########################################################################
# split and reweight a dataset

def prepare_for_merge(path, fractions):
    """ Before merging: Split -> to HDF5 -> Reweight """

    if not len(fractions) == 3:
        raise ValueError('Must have exactly 3 fractions')

    outpath_base = utils.ensure_suffix(
        string=os.path.basename(path),
        suffix='.root'
    )

    tmpdir = tempfile.mkdtemp()
    outpaths = []
    for dset in ['.training.root', '.validation.root', '.test.root']:
        outpaths.append(
            tmpdir + '/' + outpath_base.replace('.root', dset)
        )

    try:
        split(
            path,
            'NNinput',
            fractions,
            outpaths
        )
        outpaths_h5 = [root_to_h5(path) for path in outpaths]
    finally:
        shutil.rmtree(tmpdir)

    reweight(outpaths_h5)

    return outpaths_h5


########################################################################
# Dataset merging into final form

def merge(paths, outputpath):
    """Merge datasets into final hdf5 file. This will create 4 groups:
        header, training, validation test. The arrays are destructured
        and the row <-> variable correspondance is contained in
        "header". The datasets are also shuffled.
    """

    outputh5 = h5.File(outputpath, 'w-')

    for dset in ['training', 'validation', 'test']:
        slice_paths = [p for p in paths if p.endswith('.'+dset+'.h5')]
        __merge(
            slice_paths,
            dset,
            outputh5,
            store_header=(dset == 'training')
        )


def __merge(paths, dsetname, outputh5, store_header=False):

    array = __load(paths)

    slices = [
        __get_slice(array, lambda n: n.startswith('I_')),
        __get_slice(array, lambda n: n.startswith('L_')),
        __get_slice(array, lambda n: n.startswith('M_'))
    ]

    if store_header:
        __store_header(
            array=array,
            input_slice=slices[0],
            label_slice=slices[1],
            metadata_slice=slices[2],
            h5_file=outputh5
        )

    array = __destructure(array).astype('float32')
    np.random.shuffle(array)  # pylint: disable=no-member
    grp = outputh5.create_group(dsetname)

    for name, islice in zip(['inputs', 'labels', 'metadata'], slices):
        grp.create_dataset(
            name,
            data=array[:, islice],
            dtype=np.float32,
            chunks=True,
            compression='lzf'
        )


def __load(paths):

    h5files = [h5.File(p, 'r') for p in paths]
    nrow = np.sum([h5f['NNinput'].shape[0] for h5f in h5files])

    dtype = h5files[0]['NNinput'][0].dtype

    array = np.empty(nrow, dtype=dtype)

    istart = 0
    for h5f in h5files:
        data = h5f['NNinput']
        istop = istart + data.shape[0]
        array[istart:istop] = np.array(data)
        istart = istop

    return array


def __get_slice(sarray, selector):
    names = __get_header(sarray)

    indices = [i for i, name in enumerate(names) if selector(name)]

    LOGGER.debug('indices: %s', str(indices))

    ilow = min(indices)
    iup = max(indices) + 1

    if not indices == range(ilow, iup):
        raise RuntimeError('selector yields non-contiguous indices')

    if iup - ilow == 1:
        return ilow
    else:
        return slice(ilow, iup)


def __store_header(array, input_slice, label_slice, metadata_slice, h5_file):

    header = __get_header(array)
    dty = h5.special_dtype(vlen=bytes)

    LOGGER.debug("storing inputs header")
    i_header = header[input_slice]
    h5_file.create_dataset(
        name='header/inputs',
        shape=(len(i_header),),
        dtype=dty,
        data=i_header,
        chunks=True,
        compression='lzf'
    )

    LOGGER.debug("storing labels header")
    l_header = header[label_slice]
    if not isinstance(label_slice, slice):
        l_header = [l_header]
    h5_file.create_dataset(
        name='header/labels',
        shape=(len(l_header),),
        dtype=dty,
        data=l_header,
        chunks=True,
        compression='lzf'
    )

    LOGGER.debug("storing metadata header")
    m_header = header[metadata_slice]
    h5_file.create_dataset(
        name='header/metadata',
        shape=(len(m_header),),
        dtype=dty,
        data=m_header,
        chunks=True,
        compression='lzf'
    )


def __destructure(structured, dtype=np.float64):
    return structured.view(dtype).reshape(structured.shape + (-1,))


def __get_header(sarray):
    return [
        t[0] for t in sorted(sarray.dtype.fields.items(), key=lambda k: k[1])
    ]


########################################################################
# Putting it all together

def create(input_paths, output_path, fractions):
    """ putting it all together """

    # make sure input_paths are absolute
    input_paths = [os.path.abspath(p) for p in input_paths]

    oldir = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    try:

        intermediate_paths = []
        for path in input_paths:
            LOGGER.info('preparing ' + path + ' for merging')
            intermediate_paths += prepare_for_merge(path, fractions)

        os.chdir(oldir)

        LOGGER.info('merging')
        merge(intermediate_paths, output_path)

    finally:
        shutil.rmtree(tmpdir)


def main():
    """ cli interface """

    LOGGER.info('starting')

    argp = argparse.ArgumentParser()
    argp.add_argument('--inputs', nargs='+', required=True)
    argp.add_argument('--output', required=True)
    argp.add_argument(
        '--fractions',
        nargs=3,
        default=[0.5, 0.25, 0.25],
        type=float
    )
    args = argp.parse_args()

    LOGGER.info('dumping configuration')
    LOGGER.info('inputs:')
    for path in args.inputs:
        LOGGER.info(path)
    LOGGER.info('output: ' + args.output)
    LOGGER.info('fractions: ' + str(args.fractions))

    create(args.inputs, args.output, args.fractions)

    LOGGER.info('done!')


if __name__ == '__main__':
    utils.main(main, 'create_dataset')
