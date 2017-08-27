""" Draw network output """
# pylint: disable=invalid-name
import h5py as h5
import ROOT
import root_numpy

from root_graph_utils import atlas_utils


path = "/lcg/storage15/atlas/gagnon/work/2017-06-19_deep-SUSY/opt_2017-08-18/42885.hades_opt_1/gen_001.03d2198c-4dbd-4819-8ff7-400b4fdbfadd_evaluated-validation.h5"  # noqa pylint: disable=line-too-long


dfile = h5.File(path, 'r')

sigkey = 'Gtt_1900_5000_1'

dset = dfile[sigkey]

keys = ['signal/' + sigkey] + \
       ['background/' + k for k in dset['background'].keys()]


ROOT.gROOT.SetBatch()
atlas_utils.set_atlas_style()

c = ROOT.TCanvas('c', '', 0, 0, 800, 600)
stk = ROOT.THStack('stk', '')
for i, k in enumerate(keys):

    hist = ROOT.TH1F(k, '', 25, 0, 1)
    root_numpy.fill_hist(hist, dset[k].value[:, 0])
    hist.Scale(1.0 / hist.Integral())
    # if not key.startswith('signal'):
    #     hist.SetLineStyle()
    stk.Add(hist)

stk.Draw('nostack')
c.SetLogy()
c.SaveAs('test.pdf')
