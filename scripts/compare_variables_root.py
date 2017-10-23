import argparse
import logging

import ROOT

from deep_susy import utils
from root_graph_utils import atlas_utils, compare_histograms


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--inputs', nargs='+', required=True)
    args.add_argument('--labels', nargs='+', required=True)
    args.add_argument('--tree', default='NNinput')
    args.add_argument('--output-prefix', default='h')
    return args.parse_args()


def _main():
    args = _get_args()

    tfiles = [ROOT.TFile(p, 'READ') for p in args.inputs]
    ttrees = [tf.Get(args.tree) for tf in tfiles]

    all_variables = []
    for tr in ttrees:
        all_variables.append(br.GetName() for br in tr.GetListOfBranches())
    logging.debug(all_variables)
    var_set = set(all_variables[0])
    for uvars in all_variables[1:]:
        var_set = var_set.intersection(set(uvars))

    logging.debug(var_set)
    for var in var_set:
        logging.info(var)
        compare_histograms.from_ttrees(
            ttrees,
            args.labels,
            var,
            args.output_prefix + '_' + var + '.pdf'
        )


if __name__ == '__main__':
    utils.main(_main, '')
