import argparse
import logging
import os

from deep_susy import model, utils


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('model')
    args.add_argument('output')
    args.add_argument('--force', action='store_true')
    return args.parse_args()


def _main():
    args = _get_args()
    if os.path.exists(args.output):
        if not args.force:
            raise RuntimeError('Output file exists: %s', args.output)
        else:
            logging.warning('Will overwrite %s', args.output)
    k_model = model.load_keras(args.model, compile=False)
    hp_dict = model.get_hyperparameters_gen_001(k_model)
    with open(args.output, 'w') as wfile:
        wfile.write(str(hp_dict))
        logging.info('wrote %s', args.output)


if __name__ == '__main__':
    utils.main(_main, '')
