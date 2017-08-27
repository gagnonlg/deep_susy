""" draw ... something ? """
# pylint: disable=invalid-name
import argparse
import ROOT
from root_graph_utils import compare_histograms

p = argparse.ArgumentParser()
p.add_argument('path')
args = p.parse_args()

tfile = ROOT.TFile(args.path, 'READ')
ttree = tfile.Get('NNinput')

# outfile = ROOT.TFile("NNinput_997a37b-REF.root", "CREATE")
# for br in list(ttree.GetListOfBranches()):
#     compare_histograms.ttree_to_hist(
#         tree=ttree,
#         varexp=br.GetName(),
#     )
# outfile.Write('', ROOT.TObject.kWriteDelete)

tfile_ref = ROOT.TFile('NNinput_997a37b-REF.root')

for hkey in tfile_ref.GetListOfKeys():
    name = '_'.join(hkey.GetName().split('_')[1:])
    hist_ref = tfile_ref.Get(hkey.GetName())
    hist_current = compare_histograms.ttree_to_hist(
        tree=ttree,
        varexp=name
    )
    compare_histograms.from_hists(
        hists=[hist_current, hist_ref],
        labels=['current', 'ref'],
        title=';{};events'.format(name),
        output='hist_{}.pdf'.format(name)
    )
