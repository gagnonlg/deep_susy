import argparse
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
    data = cursor.fetchall()

    dtype = [(n, 'f4') for n in columns]
    logging.debug(dtype)
    s_results = np.zeros((len(data),), dtype=dtype)

    for i, col in enumerate(columns):
        if col == 'normalization':
            i_norm = i
            break

    for i, row in enumerate(data):
        lrow = list(row)
        lrow[i_norm] = {
            'None': 0,
            '4vec': 1,
            '1402.4735': 2
        }[row[i_norm]]
        nrow = np.array([float(e) for e in lrow], dtype=np.float32)
        s_results[i] = nrow
        logging.debug(s_results[i])

    return s_results


def _colname(var):
    return {
        'n_hidden_layers': 'N. hidden layers',
        'n_hidden_units': 'N. hidden units / hidden layer',
        'normalization': 'Normalization scheme',
        'l2': 'L2',
        'n_excluded_training': 'N. excluded (training)',
        'n_excluded_validation': 'N. excluded (validation)'
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


def _graph(data, xvar, yvar, prefix):
    data = data[[xvar, yvar]]
    graph = ROOT.TGraph(data.shape[0])
    root_numpy.fill_graph(
        graph,
        dataset.destructure_array(data, indtype=np.float32)
    )
    cnv = ROOT.TCanvas('c', '', 0, 0, 800, 600)

    mgr = ROOT.TMultiGraph()
    mgr.Add(graph)
    mgr.SetMaximum(1.1 * graph.GetYaxis().GetXmax())

    mgr.SetTitle(';{};{}'.format(_colname(xvar), _colname(yvar)))
    mgr.Draw("AP")
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
        n for (n, _) in data.dtype.descr if not n.startswith('n_excluded_')
    ]

    for var in xcols:
        _graph(
            data,
            var,
            'n_excluded_validation',
            os.path.basename(args.database).replace('.hyperparameters', '')
        )

if __name__ == '__main__':
    utils.main(_main, '')
