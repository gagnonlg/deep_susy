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

    if not output.endswith('.h5') and not output.endswith('.hdf5'):
        output += '.h5'

    h5_file = h5.File(output, 'w-')

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


def __main():

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


def __get_slice(sarray, selector):
    names = [
        t[0] for t in sorted(sarray.dtype.fields.items(), key=lambda k: k[1])
    ]

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