import argparse
import logging

import h5py as h5

from deep_susy import utils


def create_group(h5file, evaluated, data, sigkey, dsetkey):
    if dsetkey == sigkey:
        grpname = 'signal'
        grpprefix = 'signal'
    else:
        grpname = dsetkey
        grpprefix = 'background'

    grp = h5file.create_group(grpname)

    input_key = '{}/{}/input'.format(grpprefix, dsetkey)
    logging.debug(input_key)
    grp.create_dataset(
        'input',
        data=data[input_key],
        compression='gzip',
     )

    metadata_key = input_key.replace('input', 'metadata')
    logging.debug(metadata_key)
    grp.create_dataset(
        'metadata',
        data=data[metadata_key],
        compression='gzip',
    )

    output_key = '{}/{}/{}'.format(sigkey, grpprefix, dsetkey)
    logging.debug(output_key)
    grp.create_dataset(
        'output',
        data=evaluated[output_key],
        compression='gzip',
    )

    return grp


def create_file(evaluated, data, sigkey):
    path = utils.unique_path(sigkey + '.h5')
    with h5.File(path, 'x') as h5file:
        for key in [sigkey] + evaluated[sigkey + '/background'].keys():
            create_group(h5file, evaluated, data, sigkey, key)
    logging.info('created ' + path)


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('evaluated')
    args.add_argument('data')
    args.add_argument('setname')
    return args.parse_args()


def main():
    args = _get_args()
    evaluated = h5.File(args.evaluated, 'r')
    data = h5.File(args.data, 'r')[args.setname]
    for sigkey in data['signal'].keys():
        logging.info(sigkey)
        create_file(evaluated, data, sigkey)


if __name__ == '__main__':
    utils.main(main, '')
