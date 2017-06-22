import argparse
import collections
import glob
import logging
import os

import ROOT

import dataset
import gtt
from root_graph_utils import atlas_utils
import utils

ROOT.gROOT.SetBatch(True)
atlas_utils.set_atlas_style()

log = logging.getLogger(__name__)

bkeys = ['WZjets', 'topEW', 'Diboson', 'ttbar', 'singletop']


def get_range(var):
    if ('_px_' in var) or ('_py_' in var):
        return 100, -2500, 2500
    if  '_pz_' in var:
        return 100, -3000, 3000
    if '_e_' in var:
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
    if 'met_mag' in var:
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

def get_hist(ddict, key, var, weighted):
    hname = 'h_{}_{}'.format(var, key)
    weight = 'M_weight' if weighted else '1.0'
    if ROOT.gDirectory.FindObject(hname) == None:
        log.debug('var: %s', var)
        nbin,vmin,vmax = get_range(var)
        rg = '({},{},{})'.format(nbin,vmin,vmax)
        varexp = '{}>>{}' + rg
    else:
        varexp = '{}>>+{}'
    for data in ddict[key]:
        data.tree.Draw(varexp.format(var, hname), weight)
    return ROOT.gDirectory.Get(hname)

def get_gtt_hist(ddict, mg, ml, var, weighted, stkint):
    dsid = str(gtt.get_dsid(mg, ml))
    weight = 'M_weight' if weighted else '1.0'
    for _dsid, tree in ddict['Gtt']:
        if dsid == _dsid:
            h_name = 'h_{}_{}'.format(var, dsid)
            nbin,vmin,vmax = get_range(var)
            rg = '({},{},{})'.format(nbin,vmin,vmax)
            varexp = '{}>>{}' + rg
            varexp = varexp.format(var, h_name)
            tree.Draw(varexp, weight)
            h = ROOT.gDirectory.Get(h_name)
            #if weighted:
            h.Scale(stkint / h.Integral())
            return h

def get_color(idist, ndist):
    #return ROOT.gStyle.GetcolorPalette(idist * 255 / ndist)
    if idist == 0:
        return ROOT.kBlack
    else:
        return ROOT.gStyle.GetColorPalette(idist * 255 / (ndist - 1))

def get_stack(ddict, var, weighted):
    log.debug('var: %s', var)
    ROOT.gStyle.SetPalette(ROOT.kBlueGreenYellow)
    stk = ROOT.THStack("stk_{}".format(var), "")
    stkint = 0
    leg = ROOT.TLegend(0.7, 0.5, 0.9, 0.9)
    hlist = []
    stkint = 0
    for i, key in enumerate(bkeys):
        h = get_hist(ddict, key, var, weighted)
        h.SetFillColor(get_color(i, len(ddict)))
        h.SetLineWidth(0)
        hlist.append(h)
        stk.Add(h)
        stkint += h.Integral()

    for h, key in zip(hlist, bkeys)[::-1]:
        leg.AddEntry(h, key, "F")

    hg = []
    hg.append(get_gtt_hist(ddict, 2100, 1, var, weighted, stkint))
    hg[-1].SetLineColor(ROOT.kGray)
    leg.AddEntry(hg[-1], 'mg=2100, ml=1', 'L')
    hg.append(get_gtt_hist(ddict, 2100, 800, var, weighted, stkint))
    hg[-1].SetLineColor(ROOT.kRed)
    leg.AddEntry(hg[-1], 'mg=2100, ml=800', 'L')
    hg.append(get_gtt_hist(ddict, 2100, 1600, var, weighted, stkint))
    hg[-1].SetLineColor(ROOT.kYellow)
    leg.AddEntry(hg[-1], 'mg=2100, ml=1600', 'L')

    leg.SetBorderSize(0)
    return stk, leg, hg

def draw_var(ddict, var, weighted):
    c = ROOT.TCanvas("c", "", 0, 0, 800, 600)
    c.SetLogy()
    stk,leg,hg = get_stack(ddict, var, weighted)
    stk.SetMaximum(stk.GetMaximum() * 2)

    stk.SetTitle(';{};{}'.format(var, 'Cross section [fb]' if weighted else 'Events'))

    stk.Draw("hist")
    for h in hg:
        h.Draw("hist same")

    leg.Draw()
    atlas_utils.atlas_label(0.2, 0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(0.325, 0.88, 'Internal simulation')

    c.SaveAs("h_{}{}.pdf".format(var, '_weighted' if weighted else ''))


def get_var_list(datasets):
    tree = datasets.itervalues().next()[0].tree
    skiplist = [
        'I_m_gluino',
        'I_m_lsp',
        'M_event_number',
        'M_run_number',
        'M_met',
        'M_dsid',
        'L_target'
    ]
    for brn in [br.GetName() for br in tree.GetListOfBranches()]:
        if brn not in skiplist:
            yield brn


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--datadir', required=True)
    return args.parse_args()

def main():

    args = get_args()

    datasets = dataset.lookup(args.datadir, 'NNinput')

    for var in get_var_list(datasets):
        # print '==> ' + var
        draw_var(datasets, var, weighted=True)
        draw_var(datasets, var, weighted=False)


if __name__ == '__main__':
    utils.main(main, 'signal_vs_bkg')
