""" Show input distributions """
# pylint: disable=invalid-name
import argparse
from itertools import izip_longest
import os
import shutil
import subprocess
import tempfile
import ROOT
from root_graph_utils import atlas_utils, hist_utils

ROOT.gROOT.SetBatch(True)
atlas_utils.set_atlas_style()

args = argparse.ArgumentParser()
args.add_argument('--input', required=True)
args.add_argument('--save-plots')
args = args.parse_args()

f_tree = ROOT.TFile(args.input)
tree = f_tree.Get("NNinput")

for evt in tree:
    dsid = str(int(evt.M_dsid))
    break

if args.save_plots is None:
    savedir = tempfile.mkdtemp()
else:
    savedir = args.save_plots
    os.mkdir(savedir)


def draw_inclusive(var):
    canvas = ROOT.TCanvas("c", "", 0, 0, 800, 600)
    tree.Draw(var+">>h_"+var)
    hist = ROOT.gDirectory.Get("h_"+var)
    hist_utils.normalize(hist)
    hist.SetMaximum(hist.GetMaximum() * 1.2)

    hist.Draw()

    hist.SetTitle(";"+var+";events")
    txt = ROOT.TText()
    txt.SetNDC()
    txt.DrawText(0.6, 0.8, "DSID: " + dsid)

    atlas_utils.atlas_label(0.2, 0.85)
    txt.DrawText(0.33, 0.85, "Internal")

    fname = savedir+"/test_inclusive_"+var+".pdf"
    canvas.SaveAs(fname)
    return fname


def draw_nonzero(varn):
    canvas = ROOT.TCanvas("c", "", 0, 0, 800, 600)

    fields = varn.split('_')
    if fields[-1].isdigit():
        condfields = fields[:]
        condfields[-2] = 'pt'
        condvar = '_'.join(condfields)
    else:
        return

    tree.Draw(varn+">>h_"+varn, condvar+">0")
    hist = ROOT.gDirectory.Get("h_"+varn)

    try:
        hist_utils.normalize(hist)
    except ZeroDivisionError:
        return

    hist.SetMaximum(hist.GetMaximum() * 1.2)

    hist.Draw()

    hist.SetTitle(";"+varn+";events")
    txt = ROOT.TText()
    txt.SetNDC()
    txt.DrawText(0.6, 0.8, "DSID: " + dsid)

    atlas_utils.atlas_label(0.2, 0.85)
    txt.DrawText(0.33, 0.85, "Internal")

    fname = savedir+"/test_nonzero_"+varn+".pdf"
    canvas.SaveAs(fname)
    return fname


def draw_01lepton(varn):
    canvas = ROOT.TCanvas("c", "", 0, 0, 800, 600)
    tree.Draw(varn+">>h_0_"+varn, "M_nlepton==0")
    tree.Draw(varn+">>h_1_"+varn, "M_nlepton>0")
    h_0 = ROOT.gDirectory.Get("h_0_"+varn)
    h_1 = ROOT.gDirectory.Get("h_1_"+varn)
    h_1.SetLineColor(ROOT.kRed)
    hist_utils.normalize(h_0)
    hist_utils.normalize(h_1)

    hstk = ROOT.THStack("hstk_01l_"+varn, "")
    hstk.Add(h_0)
    hstk.Add(h_1)

    hstk.SetMaximum(hstk.GetMaximum("nostack") * 1.2)
    hstk.SetTitle(";"+varn+";events")

    hstk.Draw("nostack")

    leg = ROOT.TLegend(0.6, 0.8, 0.9, 0.9)
    leg.SetBorderSize(0)
    leg.AddEntry(h_0, "==0 lepton", "L")
    leg.AddEntry(h_1, ">=1 lepton", "L")
    leg.Draw()

    txt = ROOT.TText()
    txt.SetNDC()
    txt.DrawText(0.6, 0.7, "DSID: " + dsid)

    atlas_utils.atlas_label(0.2, 0.85)
    txt.DrawText(0.33, 0.85, "Internal")

    fname = savedir+"/test_01lepton_"+varn+".pdf"
    canvas.SaveAs(fname)
    return fname

f_incl = []
f_nonz = []
f_01le = []

for varn in [v.GetName() for v in tree.GetListOfBranches()]:
    f_incl.append(draw_inclusive(varn))
    f_nonz.append(draw_nonzero(varn))
    f_01le.append(draw_01lepton(varn))


def draw_cutflow(weighted):
    """ draw the cutflow """
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
    txt = ROOT.TText()
    txt.SetNDC()
    txt.DrawText(0.65, 0.78, "DSID: " + dsid)
    txt.DrawText(0.65, 0.71, "Weighted" if weighted else "Unweighted")
    atlas_utils.atlas_label(0.65, 0.85)
    txt.DrawText(0.78, 0.85, "Internal")
    fname = savedir+"/cutflow{}.pdf".format("_weighted" if weighted else "")
    canvas.SaveAs(fname)
    return fname

f_cut = []
f_cut.append(draw_cutflow(weighted=True))
f_cut.append(draw_cutflow(weighted=False))

########################################################################
# report

doc = ""


def frame1(title, topleft):
    """ 1 figure frame """
    global doc  # pylint: disable=global-statement
    doc += "\\begin{frame}\n"
    doc += "\\frametitle{"+title+"}\n"
    doc += "\\begin{columns}\n"
    doc += "\\begin{column}{0.5\\textwidth}\n"
    doc += "\\includegraphics[width=\\textwidth]{"+topleft+"}\n"
    doc += "\\end{column}\n"
    doc += "\\end{columns}\n"
    doc += "\\end{frame}\n"


def frame2(title, topleft, topright):
    """ 2 figure frame """
    global doc  # pylint: disable=global-statement
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
    """ 3 figure frame """
    global doc  # pylint: disable=global-statement
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
    """ 4 figure frame """
    global doc  # pylint: disable=global-statement
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
    """ 1, 2, 3 or 4 figure frame """
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


def grouper(iterable, glen, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    return izip_longest(fillvalue=fillvalue, *([iter(iterable)] * glen))

for paths in grouper(f_incl, 4):
    frame("Inclusive", *paths)
for paths in grouper(f_nonz, 4):
    frame("Non-zero objects only", *paths)
for paths in grouper(f_01le, 4):
    frame("0 + 1 lepton channels", *paths)

frame2("cutflow", *f_cut)  # pylint: disable=no-value-for-parameter

doc += "\\end{document}\n"

path = '{}_after_preselection_report.tex'.format(dsid)

tex = open(path, 'w')
tex.write(doc)
tex.close()

rmlist = [path.replace('tex', s) for s in
          ['aux', 'log', 'nav', 'out', 'snm', 'toc']]

subprocess.call('pdflatex ' + path, shell=True)
for p in rmlist:
    os.remove(p)

if args.save_plots is None:
    shutil.rmtree(savedir)
