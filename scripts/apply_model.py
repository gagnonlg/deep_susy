""" apply a model on a dataset """
import argparse
import logging

import h5py as h5

from deep_susy import dataset, model, utils


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('model')
    args.add_argument('data')
    args.add_argument('setname')
    args.add_argument('--output')
    return args.parse_args()


def _get_masses(key):
    # 'Gtt_<GLUINO>_5000_<LSP>
    fields = key.split('_')
    return int(fields[1]), int(fields[3])


def _main():
    # pylint: disable=too-many-locals
    args = _get_args()
    k_model = model.load_keras(args.model)
    if args.output is None:
        output = args.model.replace('trained', 'evaluated-' + args.setname)
    else:
        output = args.output

    with h5.File(args.data, 'r')as infile, \
         h5.File(output, 'w') as outfile:  # noqa

        indata = infile[args.setname]

        for key in indata['signal'].keys():
            logging.info(key)
            grp = outfile.create_group(key)
            m_g, m_l = _get_masses(key)
            all_keys = ['signal/' + key] + \
                       ['background/' + k for k in indata['background'].keys()]

            for this_key in all_keys:
                logging.info('  %s', this_key)
                h5data = indata[this_key + '/input'].value
                if h5data.shape[0] == 0:
                    logging.warning('empty dataset: %s', this_key)
                else:
                    logging.debug('destructuring...')
                    data = dataset.destructure_array(h5data)
                    logging.debug('reparametrizing...')
                    data[:, -2] = m_g
                    data[:, -1] = m_l
                    logging.debug('predicting...')
                    predicted = k_model.predict(data)
                    logging.debug('writing...')
                    grp.create_dataset(
                        name=this_key,
                        data=predicted
                    )
                    logging.debug('... done.')


if __name__ == '__main__':
    utils.main(_main, '')
