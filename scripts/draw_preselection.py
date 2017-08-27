import argparse
import ROOT
from root_graph_utils import atlas_utils

ROOT.gROOT.SetBatch(True)
atlas_utils.set_atlas_style()

args = argparse.ArgumentParser()
args.add_argument('--inputs-before', nargs='+', required=True)
args.add_argument('--input-after', required=True)
args.add_argument('--dsid', required=True)
args = args.parse_args()

tree_before = ROOT.TChain("nominal")
for path in args.inputs_before:
    tree_before.Add(path)

f_tree_after = ROOT.TFile(args.input_after)
tree_after = f_tree_after.Get("nominal")


def draw_stacked(var):
    canvas = ROOT.TCanvas("c", "", 0, 0, 800, 600)
    tree_before.Draw(var+">>h_b_"+var)
    tree_after.Draw(var+">>h_a_"+var)
    h_b = ROOT.gDirectory.Get("h_b_"+var)
    h_b.Scale(1.0/h_b.Integral())
    h_a = ROOT.gDirectory.Get("h_a_"+var)
    h_a.SetLineColor(ROOT.kRed)
    h_a.Scale(1.0/h_a.Integral())
    hs = ROOT.THStack("hs_"+var, "")
    hs.Add(h_b)
    hs.Add(h_a)

    hs.SetMaximum(hs.GetMaximum("nostack") * 1.2)

    hs.Draw("nostack")

    title = args.input_after
    hs.SetTitle(";"+var+";events")

    leg = ROOT.TLegend(0.6, 0.8, 0.9, 0.9)
    leg.SetBorderSize(0)
    leg.AddEntry(h_b, "before pre-selection", "L")
    leg.AddEntry(h_b, "after pre-selection", "L")
    leg.Draw()

    txt = ROOT.TText()
    txt.SetNDC()
    txt.DrawText(0.6, 0.7, "DSID: " + args.dsid)

    atlas_utils.atlas_label(0.2, 0.85)
    txt.DrawText(0.33, 0.85, "Internal")

    canvas.SaveAs("test_"+var+".pdf")

var_before = {v.GetName() for v in tree_before.GetListOfBranches()}
var_after = {v.GetName() for v in tree_after.GetListOfBranches()}

for var in var_before.intersection(var_after):
    print "==> " + var
    draw_stacked(var)
