""" Function to assemble the dataset """
# pylint: disable=no-member
import argparse
import glob
import logging
import os
import shutil
import tempfile

import h5py as h5
import numpy as np
import root_numpy

import dataset
import utils

LOGGER = logging.getLogger('dataset.create')


def create(file_list, output, njobs=1):
    """ create the dataset.

    Args:
      file_list: list of paths to input ROOT files
      output: desired path for output hdf5 file
      njobs: number of threads to use to create the dataset
    Returns:
      None
    Throws:
      IOError: output already exists
    """

    LOGGER.info('Creating the dataset from %d input files', len(file_list))

    output = utils.ensure_suffix(output, '.h5', alt=['.hdf5'])
    h5_file = h5.File(
        name=output,
        mode='w-'  # Create file, fail if exists
    )

    try:

        tmpdir = tempfile.mkdtemp()
        os.mkdir('{}/training'.format(tmpdir))
        os.mkdir('{}/validation'.format(tmpdir))
        os.mkdir('{}/test'.format(tmpdir))

        dataset.parallel_split(
            inpaths=file_list,
            treename='NNinput',
            fractions=[0.5, 0.25, 0.25],
            nthreads=njobs,
            outdirs=['{}/{}'.format(tmpdir, name) for name in
                     ['training', 'validation', 'test']]
        )

        for group in ['training', 'validation', 'test']:

            LOGGER.info('assembling the %s set', group)

            array = root_numpy.root2array(
                glob.glob('{}/{}/*'.format(tmpdir, group))
            )

            input_slice = __get_slice(array, lambda n: n.startswith('I_'))
            label_slice = __get_slice(array, lambda n: n.startswith('L_'))
            metadata_slice = __get_slice(array, lambda n: n.startswith('M_'))
            parameter_slice = __get_slice(
                array,
                lambda n: n in ['I_m_gluino', 'I_m_lsp']
            )

            # only store header once
            if group == 'training':

                __store_header(
                    array=array,
                    input_slice=input_slice,
                    label_slice=label_slice,
                    metadata_slice=metadata_slice,
                    h5_file=h5_file
                )

            array = __destructure(array)
            __parametrize(array, label_slice, parameter_slice)
            np.random.shuffle(array)
            grp = h5_file.create_group(group)

            slices = [input_slice, label_slice, metadata_slice]
            for name, islice in zip(['inputs', 'labels', 'metadata'], slices):
                grp.create_dataset(
                    name,
                    data=array[:, islice],
                    dtype=np.float32,
                    chunks=True,
                    compression='lzf'
                )

    finally:

        shutil.rmtree(tmpdir)

    h5_file.close()
    logging.info('dataset created: %s', output)


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


def create_main():
    """ cli interface """
    args = argparse.ArgumentParser()
    args.add_argument('--inputs', nargs='+', required=True)
    args.add_argument('--output', required=True)
    args.add_argument('--njobs', type=int, default=1)
    args = args.parse_args()

    create(args.inputs, args.output, args.njobs)

    return 0


def __destructure(structured, dtype=np.float64):
    return structured.view(dtype).reshape(structured.shape + (-1,))


def __parametrize(array, target_index, parameter_slice):
    data = array[np.where(array[:, target_index] == 1)]
    data = data[:, parameter_slice]
    indices = np.where(array[:, target_index] == 0) + (parameter_slice,)

    choices = np.random.randint(data.shape[0], size=indices[0].shape)
    array[indices] = data[choices]


def __get_header(sarray):
    return [
        t[0] for t in sorted(sarray.dtype.fields.items(), key=lambda k: k[1])
    ]


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
