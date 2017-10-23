import logging

import numpy as np
import ROOT
import root_numpy

from root_graph_utils import atlas_utils

def evaluate(k_model, history, data):
    """ Trained model evaluation """
    ROOT.gROOT.SetBatch(True)
    atlas_utils.set_atlas_style()
    _fit_history_curves(history)


def _fit_history_curves(history):
    logging.info('Creating the fit history curves')

    def _enumerate(key):
        data = history[key].value
        buf = np.empty((data.shape[0], 2))
        buf[:, 0] = np.arange(data.shape[0])
        buf[:, 1] = data
        return buf

    for key in ['acc', 'loss']:
        graphs = ROOT.TMultiGraph('mg_' + key, '')
        data = _enumerate(key)
        val_data = _enumerate('val_' + key)
        graph = ROOT.TGraph(data.shape[0])
        val_graph = ROOT.TGraph(val_data.shape[0])
        root_numpy.fill_graph(graph, data)
        root_numpy.fill_graph(val_graph, val_data)
        val_graph.SetLineColor(ROOT.kRed)
        graphs.Add(graph)
        graphs.Add(val_graph)

        graph.SetLineWidth(2)
        val_graph.SetLineWidth(2)

        canvas = ROOT.TCanvas('fit_history', '', 0, 0, 800, 600)
        graphs.SetTitle(';Epoch;' + key)
        graphs.Draw('AL')
        canvas.SaveAs('fit_history_{}.pdf'.format(key))
