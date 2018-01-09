import argparse
import array
import logging
import os
import sqlite3

import numpy as np
import ROOT
import root_numpy

from deep_susy import dataset, utils
from root_graph_utils import atlas_utils


def _get_data(database):
    cursor = database.cursor()
    query = open(
        utils.project_path('sql/SELECT_FROM_perf.sql'),
        'r'
    ).read().strip()
    cursor.execute(query)
    columns = [t[0] for t in cursor.description]
    logging.debug(columns)
    data = cursor.fetchall()

    dtype = [(n, 'f4') for n in columns]
    logging.debug(dtype)
    s_results = np.zeros((len(data),), dtype=dtype)

    for i, col in enumerate(columns):
        if col == 'NORMALIZATION':
            i_norm = i
        if col == 'PARAMETRIZATION':
            i_param = i

    for i, row in enumerate(data):
        lrow = list(row)
        # TODO modify the names below
        lrow[i_norm] = {
            'None': 0,
            'normalization': 1,
            'standardize': 2
        }[row[i_norm]]
        lrow[i_param] = {
            'pxpypze': 0,
            'ptetaphim': 1
        }[row[i_param]]
        nrow = np.array([float(e) for e in lrow], dtype=np.float32)
        s_results[i] = tuple(nrow)
        logging.debug(s_results[i])

    return s_results


def _colname(var):
    return {
        'PARAMETRIZATION': 'Input parametrization',
        'NLAYERS': 'N. hidden layers',
        'NUNITS': 'N. hidden units / hidden layer',
        'NORMALIZATION': 'Normalization scheme',
        'HIDDEN_L1': 'Hidden L1',
        'HIDDEN_L2': 'Hidden L2',
        'OUTPUT_L1': 'Output L1',
        'OUTPUT_L2': 'Output L2',
        'LEARNING_RATE': 'Learning rate',
        'BATCH_NORM': 'Batch normalization',
        'DROPOUT_INPUT': 'Dropout rate (input)',
        'DROPOUT_HIDDEN': 'Dropout rate (hidden)',
        'BATCH_SIZE': 'Batch size',
        'N_EXCLUDED': 'N. excluded',
        'N_EXCLUDED_ABOVE_MBJ': 'N. excluded above MBJ',
    }[var]


def _draw_atlas_label(preliminary):
    atlas_utils.atlas_label(0.2, 0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(
        0.33,
        0.88,
        'Simulation {}'.format(
            'Preliminary' if preliminary else 'Internal'
        )
    )

def _xbins(var):
    if var == 'PARAMETRIZATION':
        b = [0, 1, 2]
    elif var in ['HIDDEN_L1', 'OUTPUT_L1', 'HIDDEN_L2', 'OUTPUT_L2']:
        b = np.linspace(0, 1e-3, 10)
    elif var == 'NLAYERS':
        b = [1, 2, 3, 4, 5, 6]
    elif var == 'NUNITS':
        b = np.linspace(100, 1000, 10)
    elif var == 'LEARNING_RATE':
        b = np.linspace(1e-4, 1e-2, 10)
    elif var == 'BATCH_NORM':
        b = [0, 1, 2]
    elif var == 'DROPOUT_INPUT':
        b = [0, 0.2, 0.4]
    elif var == 'DROPOUT_HIDDEN':
        b = [0, 0.5, 1.0]
    elif var == 'BATCH_SIZE':
        b = [32, 64, 128, 256, 512]
    elif var == 'NORMALIZATION':
        b = [0, 1, 2, 3]

    return array.array('d', b)


def _ybins(var):
    if var == 'N_EXCLUDED':
        b = np.arange(0, 101)
    elif var == 'N_EXCLUDED_ABOVE_MBJ':
        b = np.arange(0, 21)
    return array.array('d', b)


def _graph(data, xvar, yvar, prefix):
    x_data = data[xvar].reshape((data.shape[0], 1))
    y_data = data[yvar].reshape((data.shape[0], 1))

    xbins = _xbins(xvar)
    ybins = _ybins(yvar)
    hist = ROOT.TH2D(
        'h_{}_{}'.format(xvar, yvar),
        '',
        len(xbins) - 1,
        xbins,
        len(ybins) - 1,
        ybins
    )
    root_numpy.fill_hist(hist, np.concatenate((x_data, y_data), axis=1))

    cnv = ROOT.TCanvas('c', '', 0, 0, 800, 600)
    hist.SetTitle(';{};{}'.format(_colname(xvar), _colname(yvar)))
    hist.Draw('CANDLE3')

    _draw_atlas_label(preliminary=False)
    cnv.SaveAs('{}_{}_{}.pdf'.format(prefix, xvar, yvar))


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('database')
    return args.parse_args()


def _main():
    args = _get_args()

    ROOT.gROOT.SetBatch()
    atlas_utils.set_atlas_style()

    db = sqlite3.connect(args.database)
    data = _get_data(db)

    xcols = [
        n for (n, _) in data.dtype.descr if not n.startswith('N_EXCLUDED')
    ]

    for var in xcols:
        # _graph(
        #     data,
        #     var,
        #     'N_EXCLUDED',
        #     os.path.basename(args.database).replace('.hyperparameters', '')
        # )
        _graph(
            data,
            var,
            'N_EXCLUDED_ABOVE_MBJ',
            os.path.basename(args.database).replace('.hyperparameters', '')
        )

if __name__ == '__main__':
    utils.main(_main, '')
