""" Program to compute expected Z across the Gtt grid """
# pylint: disable=no-member
import argparse
import array
import itertools
import logging
import os

import h5py as h5
import numpy as np
import ROOT
import root_numpy

from deep_susy import utils
from root_graph_utils import atlas_utils


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


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('evaluated')

    # to get the weights
    args.add_argument('data')
    args.add_argument('setname')

    args.add_argument('--lumi', default=36.1, type=float)
    args.add_argument('--uncert', default=0.3, type=float)
    args.add_argument('--MBJ')

    return args.parse_args()


def _bkg_keys(dfile, sigkey):

    for key in dfile[sigkey + '/background']:
        if dfile[sigkey + '/background/' + key].shape[0] > 0:
            yield key


def _threshold(dfile, sigkey, metadata):

    ttbar = None
    for k in _bkg_keys(dfile, sigkey):
        if 'ttbar' in k:
            ttbar = k
            break

    weights = metadata['background/' + ttbar + '/metadata'].value['M_weight']
    scores = dfile[sigkey + '/background/' + ttbar].value[:, 0]

    isort = np.argsort(scores)[::-1]
    scores = scores[isort]
    weights = weights[isort]

    wsum = np.cumsum(weights)
    wsum2 = np.sqrt(np.cumsum(weights * weights))
    funcert = wsum2 / wsum

    imin = np.argmin(np.abs(funcert - 0.3))

    return scores[imin]


def _yield(scores, weights, threshold):
    logging.debug('scores: %s, weights: %s', scores.shape, weights.shape)
    return np.sum(weights[np.where(scores[:, 0] > threshold)])


def _main():
    args = _get_args()
    dfile = h5.File(args.evaluated, 'r')
    data = h5.File(args.data, 'r')[args.setname]

    if os.path.exists('CONTOUR_CACHE.txt'):
        logging.info('Loading cached data')
        _make_contour(
            np.loadtxt(
                'CONTOUR_CACHE.txt',
                dtype=[('mg', 'i4'), ('ml', 'i4'), ('z', 'f4')]
            ),
            os.path.basename(args.evaluated).replace('.h5', '') + '.pdf',
            args.MBJ
        )
        return

    results = np.zeros(
        len(dfile.keys()),
        dtype=[('mg', 'i4'), ('ml', 'i4'), ('z', 'f4')]
    )

    for i, sigkey in enumerate(dfile.keys()):
        thr = _threshold(dfile, sigkey, data)
        s_yield = _yield(
            scores=dfile[sigkey + '/signal/' + sigkey].value,
            weights=data['signal/' + sigkey + '/metadata'].value['M_weight'],
            threshold=thr
        )
        b_yield = 0
        for bkgkey in _bkg_keys(dfile, sigkey):
            logging.debug('  %s', bkgkey)
            b_yield += _yield(
                scores=dfile[sigkey + '/background/' + bkgkey].value,
                weights=data['background/' + bkgkey + '/metadata'].value[
                    'M_weight'
                ],
                threshold=thr
            )
        expz = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(
            s_yield * args.lumi,
            b_yield * args.lumi,
            args.uncert
        )

        fields = sigkey.split('_')
        results[i] = (int(fields[1]), int(fields[3]), expz)
        logging.info('%s: %f', sigkey, expz)

    excluded = results[np.where(results['z'] >= 1.64)]
    max_m = (np.max(excluded['mg']), np.max(excluded['ml']))
    n_excluded = excluded.shape[0]

    logging.info(
        'reach: m_gluino = %s, m_lsp = %s, n_exluded = %s',
        max_m[0],
        max_m[1],
        n_excluded
    )
    open('/dev/stdout', 'w').write(
        "{} {} {}\n".format(max_m[0], max_m[1], n_excluded)
    )

    np.savetxt('CONTOUR_CACHE.txt', results)

    _make_contour(
        results,
        os.path.basename(args.evaluated).replace('.h5', '') + '.pdf',
        args.MBJ
    )


def _make_contour(results, path, mbj):

    ROOT.gROOT.SetBatch()
    atlas_utils.set_atlas_style()
    ROOT.gStyle.SetPalette(ROOT.kBird)

    bins_x = [
        900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
        1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500
    ]
    bins_y = [
        1, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800,
        2000, 2200
    ]

    bin_data = np.zeros((len(bins_x) * len(bins_y), 3))
    for i, (m_g, m_l) in enumerate(itertools.product(bins_x, bins_y)):
        bin_data[i, 0] = m_g
        bin_data[i, 1] = m_l

        isel = np.where(
            np.logical_and(results['mg'] == m_g, results['ml'] == m_l)
        )[0]
        if isel.shape[0] == 1:
            bin_data[i, 2] = max([results['z'][isel], 0])
        elif isel.shape[0] > 1:
            raise RuntimeError("ambiguous mg, ml")

    hist = ROOT.TH2F(
        'contour',
        '',
        len(bins_x) - 1,
        array.array('f', bins_x),
        len(bins_y) - 1,
        array.array('f', bins_y)
    )
    root_numpy.fill_hist(hist, bin_data[:, :2], bin_data[:, 2])

    ROOT.gStyle.SetPadRightMargin(0.15)
    cnv = ROOT.TCanvas('c', '', 0, 0, 800, 600)
    hist_contour = hist.Clone()
    hist.SetTitle(';m_{#tilde{g}};m_{#tilde{#chi}^{0}_{1}}')
    hist.GetZaxis().SetTitle('Expected significance')
    hist.GetYaxis().SetRangeUser(1, 2300)
    ROOT.gStyle.SetPaintTextFormat(".2f")
    hist.Draw('COLZ TEXT')
    hist_contour.SetLineWidth(3)
    hist_contour.SetLineStyle(1)
    hist_contour.SetLineColor(ROOT.kRed)
    hist_contour.SetContour(1)
    hist_contour.SetContourLevel(0, 1.64)
    hist_contour.Draw('CONT3 same')

    leg = ROOT.TLegend(0.63, 0.8, 0.83, 0.92)
    leg.AddEntry(hist_contour, '2#sigma exclusion', 'L')

    if mbj:
        ifile = ROOT.TFile(mbj, 'READ')
        mbj = ifile.Get('contour')
        mbj.SetLineWidth(3)
        mbj.SetLineStyle(2)
        mbj.SetLineColor(ROOT.kRed)
        mbj.SetContour(1)
        mbj.SetContourLevel(0, 1.64)
        mbj.Draw('CONT3 same')
        leg.AddEntry(mbj, 'MBJ 2#sigma exclusion', 'L')

    leg.SetBorderSize(0)
    leg.Draw()

    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.SetTextSize(txt.GetTextSize() * 0.65)
    txt.DrawLatex(
        0.2,
        0.82,
        '#tilde{g}#rightarrow t #bar{t} + #tilde{#chi}^{0}_{1}'
    )
    txt.DrawText(
        0.2,
        0.77,
        'model: {}'.format(path.split('_evaluated')[0])
    )

    _draw_atlas_label(preliminary=False)

    cnv.SaveAs(path)


if __name__ == '__main__':
    utils.main(_main, '')
