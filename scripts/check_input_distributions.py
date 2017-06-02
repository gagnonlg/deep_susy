import argparse
from itertools import izip_longest
import os
import shutil
import subprocess
import tempfile
import ROOT
from root_graph_utils import atlas_utils, hist_utils

ROOT.gROOT.SetBatch(True)
atlas_utils.set_atlas_style();

args = argparse.ArgumentParser()
args.add_argument('--input', required=True)
args.add_argument('--dsid', required=True)
args = args.parse_args()

f_tree = ROOT.TFile(args.input)
tree = f_tree.Get("NNinput")

tmpdir = tempfile.mkdtemp()

def draw_inclusive(var):
    canvas = ROOT.TCanvas("c", "", 0, 0, 800, 600)
    tree.Draw(var+">>h_"+var)
    h = ROOT.gDirectory.Get("h_"+var)
    hist_utils.normalize(h)
    h.SetMaximum(h.GetMaximum() * 1.2)

    h.Draw()

    title = args.input
    h.SetTitle(";"+var+";events")
    txt = ROOT.TText();
    txt.SetNDC()
    txt.DrawText(0.6, 0.8, "DSID: " + args.dsid)

    atlas_utils.atlas_label(0.2, 0.85)
    txt.DrawText(0.33, 0.85, "Internal")

    fname = tmpdir+"/test_inclusive_"+var+".pdf"
    canvas.SaveAs(fname)
    return fname

def draw_nonzero(var):
    canvas = ROOT.TCanvas("c", "", 0, 0, 800, 600)

    fields = var.split('_')
    if fields[-1].isdigit():
        condfields = fields[:]
        condfields[-2] = 'pt'
        condvar = '_'.join(condfields)
    else:
        return

    tree.Draw(var+">>h_"+var, condvar+">0")
    h = ROOT.gDirectory.Get("h_"+var)

    try:
        hist_utils.normalize(h)
    except ZeroDivisionError:
        return

    h.SetMaximum(h.GetMaximum() * 1.2)

    h.Draw()

    title = args.input
    h.SetTitle(";"+var+";events")
    txt = ROOT.TText();
    txt.SetNDC()
    txt.DrawText(0.6, 0.8, "DSID: " + args.dsid)

    atlas_utils.atlas_label(0.2, 0.85)
    txt.DrawText(0.33, 0.85, "Internal")

    fname = tmpdir+"/test_nonzero_"+var+".pdf"
    canvas.SaveAs(fname)
    return fname



def draw_01lepton(var):
    canvas = ROOT.TCanvas("c", "", 0, 0, 800, 600)
    tree.Draw(var+">>h_0_"+var, "M_nlepton==0")
    tree.Draw(var+">>h_1_"+var, "M_nlepton>0")
    h0 = ROOT.gDirectory.Get("h_0_"+var)
    h1 = ROOT.gDirectory.Get("h_1_"+var)
    h1.SetLineColor(ROOT.kRed)
    hist_utils.normalize(h0)
    hist_utils.normalize(h1)

    hs = ROOT.THStack("hs_01l_"+var, "")
    hs.Add(h0)
    hs.Add(h1)

    hs.SetMaximum(hs.GetMaximum("nostack") * 1.2)
    hs.SetTitle(";"+var+";events")

    hs.Draw("nostack")

    leg = ROOT.TLegend(0.6, 0.8, 0.9, 0.9)
    leg.SetBorderSize(0)
    leg.AddEntry(h0, "==0 lepton", "L")
    leg.AddEntry(h1, ">=1 lepton", "L")
    leg.Draw()

    title = args.input
    txt = ROOT.TText();
    txt.SetNDC()
    txt.DrawText(0.6, 0.7, "DSID: " + args.dsid)

    atlas_utils.atlas_label(0.2, 0.85)
    txt.DrawText(0.33, 0.85, "Internal")

    fname = tmpdir+"/test_01lepton_"+var+".pdf"
    canvas.SaveAs(fname)
    return fname



vars = [v.GetName() for v in tree.GetListOfBranches()]

f_incl = []
f_nonz = []
f_01le = []

for var in vars:
    f_incl.append(draw_inclusive(var))
    f_nonz.append(draw_nonzero(var))
    f_01le.append(draw_01lepton(var))

def draw_cutflow(weighted):
    canvas = ROOT.TCanvas("c", "", 0, 0, 800, 600)
    cutflow = f_tree.Get("cutflow" + ("_weighted" if weighted else ""))
    cutflow.Scale(1.0 / cutflow.GetBinContent(1))
    cutflow.GetXaxis().SetBinLabel(1, "Initial")
    cutflow.GetXaxis().SetBinLabel(2, "MET trigger")
    cutflow.GetXaxis().SetBinLabel(3, "MET > 200")
    cutflow.GetXaxis().SetBinLabel(4, "njet30 >= 4")
    cutflow.GetXaxis().SetBinLabel(5, "nbjet77 >= 2")
    cutflow.GetXaxis().SetBinLabel(6, "nlepton== 0 && dphimin > 0.4")
    cutflow.GetXaxis().SetBinLabel(7, "met_filter")
    cutflow.GetXaxis().SetBinLabel(8, "ht_filter")
    ROOT.gStyle.SetPaintTextFormat(".2f")
    cutflow.SetTitle(";;efficiency")
    cutflow.Draw("HIST TEXT00")
    txt = ROOT.TText();
    txt.SetNDC()
    txt.DrawText(0.65, 0.78, "DSID: " + args.dsid)
    txt.DrawText(0.65, 0.71, "Weighted" if weighted else "Unweighted")
    atlas_utils.atlas_label(0.65, 0.85)
    txt.DrawText(0.78, 0.85, "Internal")
    fname = tmpdir+"/cutflow{}.pdf".format("_weighted" if weighted else "")
    canvas.SaveAs(fname)
    return fname

f_cut = []
f_cut.append(draw_cutflow(weighted=True))
f_cut.append(draw_cutflow(weighted=False))

########################################################################
#### report

doc = ""

def frame1(title, topleft):
    global doc
    doc += "\\begin{frame}\n"
    doc += "\\frametitle{"+title+"}\n"
    doc += "\\begin{columns}\n"
    doc += "\\begin{column}{0.5\\textwidth}\n"
    doc += "\\includegraphics[width=\\textwidth]{"+topleft+"}\n"
    doc += "\\end{column}\n"
    doc += "\\end{columns}\n"
    doc += "\\end{frame}\n"


def frame2(title, topleft, topright):
    global doc
    doc += "\\begin{frame}\n"
    doc += "\\frametitle{"+title+"}\n"
    doc += "\\begin{columns}\n"
    doc += "\\begin{column}{0.5\\textwidth}\n"
    doc += "\\includegraphics[width=\\textwidth]{"+topleft+"}\n"
    doc += "\\end{column}\n"
    doc += "\\begin{column}{0.5\\textwidth}\n"
    doc += "\\includegraphics[width=\\textwidth]{"+topright+"}\n"
    doc += "\\end{column}\n"
    doc += "\\end{columns}\n"
    doc += "\\end{frame}\n"

def frame3(title, topleft, topright, botleft):
    global doc
    doc += "\\begin{frame}\n"
    doc += "\\frametitle{"+title+"}\n"
    doc += "\\begin{columns}\n"
    doc += "\\begin{column}{0.5\\textwidth}\n"
    doc += "\\includegraphics[width=\\textwidth]{"+topleft+"}\\\\\n"
    doc += "\\includegraphics[width=\\textwidth]{"+botleft+"}\n"
    doc += "\\end{column}\n"
    doc += "\\begin{column}{0.5\\textwidth}\n"
    doc += "\\includegraphics[width=\\textwidth]{"+topright+"}\n"
    doc += "\\end{column}\n"
    doc += "\\end{columns}\n"
    doc += "\\end{frame}\n"


def frame4(title, topleft, topright, botleft, botright):
    global doc
    doc += "\\begin{frame}\n"
    doc += "\\frametitle{"+title+"}\n"
    doc += "\\begin{columns}\n"
    doc += "\\begin{column}{0.5\\textwidth}\n"
    doc += "\\includegraphics[width=\\textwidth]{"+topleft+"}\\\\\n"
    doc += "\\includegraphics[width=\\textwidth]{"+botleft+"}\n"
    doc += "\\end{column}\n"
    doc += "\\begin{column}{0.5\\textwidth}\n"
    doc += "\\includegraphics[width=\\textwidth]{"+topright+"}\\\\\n"
    doc += "\\includegraphics[width=\\textwidth]{"+botright+"}\n"
    doc += "\\end{column}\n"
    doc += "\\end{columns}\n"
    doc += "\\end{frame}\n"

def frame(title, topleft, topright, botleft, botright):
    if topleft is None:
        return
    elif topright is None:
        frame1(title, topleft)
    elif botleft is None:
        frame2(title, topleft, topright)
    elif botright is None:
        frame3(title, topleft, topright, botleft)
    else:
        frame4(title, topleft, topright, botleft, botright)

doc += "\\documentclass{beamer}\n\\usepackage{graphicx}\n\\begin{document}\n"


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

for paths in grouper(f_incl, 4):
    frame("Inclusive", *paths)
for paths in grouper(f_nonz, 4):
    frame("Non-zero objects only", *paths)
for paths in grouper(f_01le, 4):
    frame("0 + 1 lepton channels", *paths)

frame2("cutflow", *f_cut)

doc += "\\end{document}\n"

path = '{}_after_preselection_report.tex'.format(args.dsid)

tex = open(path, 'w')
tex.write(doc)
tex.close()

rmlist = [path.replace('tex', s) for s in
          ['aux', 'log', 'nav', 'out', 'snm', 'toc']]

subprocess.call('pdflatex ' + path, shell=True)
for p in rmlist:
    os.remove(p)
shutil.rmtree(tmpdir)
