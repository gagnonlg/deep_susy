import dataset

import ROOT

from root_graph_utils import atlas_utils

ROOT.gROOT.SetBatch(True)
atlas_utils.set_atlas_style()
ROOT.gStyle.SetPalette(ROOT.kRainBow)

ddict = dataset.lookup(
    datadir='/lcg/storage15/atlas/gagnon/data/deep_susy/multib_2.4.28',
    treename='nominal',
    xsec=True
)


def get_weight_list(ddict):
    tree = ddict.itervalues().next()[0].tree
    for br in tree.GetListOfBranches():
        name = br.GetName()
        if name.startswith('weight'):
            yield name

print list(get_weight_list(ddict))

for cpn in ddict:
    c = ROOT.TCanvas('c', '', 0, 0, 800, 600)
    stk = ROOT.THStack('stk', '')
    leg = ROOT.TLegend(0.2, 0.4, 0.5, 0.79)
    for i, var in enumerate(get_weight_list(ddict)):
        hname = 'h_{}_{}'.format(cpn, var)
        for dset in ddict[cpn]:
            if ROOT.gDirectory.FindObject(hname) == None:  # noqa
                varexp = '{}>>{}(100,-2,2)'
            else:
                varexp = '{}>>+{}'
        dset.tree.Draw(varexp.format(var, hname))
        hist = ROOT.gDirectory.Get(hname)
        hist.SetLineColor(ROOT.gStyle.GetColorPalette(i * 255 / 11))
        leg.AddEntry(hist, var, 'L')
        stk.Add(hist)
    stk.SetTitle(';Weight;Events')
    stk.Draw('nostack')
    atlas_utils.atlas_label(0.2, 0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(0.325, 0.88, 'Internal simulation')
    txt.DrawText(0.2, 0.82, cpn)
    leg.SetBorderSize(0)
    leg.Draw()
    c.SaveAs('weights_{}.pdf'.format(cpn))

for i, cpn in enumerate(ddict):
    c = ROOT.TCanvas('ca'+cpn, '', 0, 0, 800, 600)
    weights = [dat.xsec for dat in ddict[cpn]]
    stats = [dat.tree.GetEntries() for dat in ddict[cpn]]
    h = ROOT.TH1F('w_xsec_' + cpn, '', 1000, 0, 1)
    print cpn
    for w, n in zip(weights, stats):
        h.Fill(w, n)
    h.GetXaxis().SetRangeUser(0, 0.1)
    h.SetTitle(';cross section #times eff weight;Events')
    h.Draw('HIST')
    atlas_utils.atlas_label(0.5, 0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(0.625, 0.88, 'Internal simulation')
    txt.DrawText(0.5, 0.82, cpn)
    c.SaveAs('weights_xsec_{}.pdf'.format(cpn))
