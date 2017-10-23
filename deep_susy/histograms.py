""" functions to draw histograms """

import itertools
import os

import ROOT
import root_numpy

from root_graph_utils import atlas_utils

atlas_utils.set_atlas_style()
ROOT.gROOT.SetBatch()
ROOT.gStyle.SetPalette(ROOT.kBlueGreenYellow)

WEIGHTS = {}


def signal_vs_background(
        signals,
        backgrounds,
        weighted,
        directory='.',
        variables='all'):
    """ Draw signal vs background histograms

    Arguments:
      -- signals: list of signal h5 groups
      -- backgrounds: list of background h5 groups
      -- weighted: if True, weight the samples to their cross-sections
      -- directory (default: '.'): output directory for the plots
      -- variables (default: 'all'): list of variables to plot or 'all'
    """

    if directory != '.':
        try:
            os.makedirs(directory)
        except OSError:
            pass

    if variables == 'all':
        variables = [
            m for m in itertools.chain(
                [('input', n) for n, _ in
                 backgrounds[0]['input'].dtype.descr],
                [('metadata', n) for n, _ in
                 backgrounds[0]['metadata'].dtype.descr],
            )
        ]
        variables.remove(('input', 'I_m_gluino'))
        variables.remove(('input', 'I_m_lsp'))
        variables.remove(('metadata', 'M_weight'))
        variables.remove(('metadata', 'M_run_number'))
        variables.remove(('metadata', 'M_event_number'))
        variables.remove(('metadata', 'M_dsid'))
    else:
        variables = [
            ('input' if v.startswith('I_') else 'metadata', v)
            for v in variables
        ]

    for dset, var in variables:
        leg = ROOT.TLegend(0.75, 0.5, 0.95, 0.9)
        bstack, bint = _stack(backgrounds, dset, var, weighted, leg)
        shists = _hists(signals, dset, var, weighted, leg, bint)
        _draw(bstack, shists, var, weighted, directory, leg)


def _get_weights(dset):
    if dset.name not in WEIGHTS:
        WEIGHTS[dset.name] = dset['metadata'].value['M_weight']
    return WEIGHTS[dset.name]


def _draw(bstack, shists, var, weighted, directory, legend):
    # pylint: disable=too-many-arguments
    cnv = ROOT.TCanvas("c", "", 0, 0, 800, 600)
    cnv.SetLogy()
    bstack.SetMaximum(bstack.GetMaximum() * 5)
    unit = _units(var)
    bstack.SetTitle(
        ';{} [{}];{}'.format(
            var,
            unit,
            '{} / {} {}'.format(
                'Cross section [fb]' if weighted else 'Events',
                shists[0].GetBinWidth(1),
                unit
            )
        )
    )
    bstack.Draw("hist")
    bstack.GetXaxis().SetLabelSize(0.04)
    for hist in shists:
        hist.Draw("hist same")

    legend.SetBorderSize(0)
    legend.Draw()
    atlas_utils.atlas_label(0.2, 0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(0.325, 0.88, 'Internal simulation')

    cnv.SaveAs(
        "{}/h_{}{}.pdf".format(
            directory if directory is not None else '.',
            var,
            '_weighted' if weighted else ''
        )
    )


def _stack(groups, dset, var, weighted, legend):
    stk = ROOT.THStack('stk_{}_{}'.format(var, weighted), '')
    icols = [0, 90, 125, 200]
    stkint = 0
    for i, grp in enumerate(groups):
        data = grp[dset].value[var]
        if weighted:
            weights = _get_weights(grp)
        else:
            weights = None
        name = grp.name.split('/')[-1]
        hist = ROOT.TH1D(
            'h_{}_{}_{}'.format(name, var, weighted),
            '',
            *_get_range(var)
        )
        col = ROOT.gStyle.GetColorPalette(icols[i])
        hist.SetLineWidth(0)
        hist.SetFillColor(col)
        legend.AddEntry(hist, name, 'F')
        root_numpy.fill_hist(hist, data, weights)
        stkint += hist.Integral()
        stk.Add(hist)
    return stk, stkint


def _hists(groups, dset, var, weighted, legend, stkint):
    # pylint: disable=too-many-arguments
    lst = []
    cols = [ROOT.kGray, ROOT.kRed, ROOT.kYellow]
    for i, grp in enumerate(groups):
        data = grp[dset].value[var]
        if weighted:
            weights = _get_weights(grp)
        else:
            weights = None
        name = grp.name.split('/')[-1]
        hist = ROOT.TH1D(
            'h_{}_{}_{}'.format(name, var, weighted),
            '',
            *_get_range(var)
        )
        hist.SetLineColor(cols[i])
        legend.AddEntry(hist, name, 'L')
        root_numpy.fill_hist(hist, data, weights)
        hist.Scale(stkint / hist.Integral())
        lst.append(hist)
    return lst


def _get_range(var):
    #  pylint: disable=too-many-return-statements,too-many-branches
    if ('_px' in var) or ('_py' in var):
        if 'lepton' in var:
            return 100, -1000, 1000
        return 100, -2500, 2500
    if '_pz' in var or '_pt' in var:
        if 'lepton' in var:
            return 100, -1500, 1500
        return 100, -3000, 3000
    if '_e_' in var:
        if 'lepton' in var:
            return 100, 0, 2000
        return 100, 000, 5000
    if '_eta_' in var:
        return 100, -3, 6
    if '_phi' in var:
        return 100, -3.5, 6.5
    if '_m_' in var:
        if 'lepton' in var:
            return 100, -0.5, 0.5
        return 100, 0, 300
    if '_isb_' in var:
        return 2, 0, 2
    if '_met' in var:
        return 100, 200, 2000
    if 'weight' in var:
        return 100, 0, 0.05
    if '_meff' in var:
        return 100, 0, 8000
    if '_mt' in var:
        return 100, 0, 2000
    if '_mjsum' in var:
        return 100, 0, 1500
    if '_nb77' in var:
        return 8, 2, 10
    if '_nlepton' in var:
        return 8, 0, 8
    if '_njet30' in var:
        return 16, 4, 20
    if '_dphimin4j' in var:
        return 100, 0, 4


def _units(var):
    if '_pt' in var or '_m' in var or '_px' in var or '_py' in var or '_pz' or '_e_' in var:
        return 'GeV'
    if '_m_gluino_' in var or '_m_lsp' in var:
        return 'GeV'
    if var in ['M_meff', 'M_mt', 'M_mtb', 'M_mjsum', 'M_met']:
        return 'GeV'
    else:
        return ''
