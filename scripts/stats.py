""" script and library to count available statistics """

import argparse
import collections
import cPickle
import logging
import functools
import resource

import h5py as h5
import numpy as np
import pylatex as latex
import ROOT

import signal_regions
import utils


LOG = logging.getLogger(__name__)


def _expected_z(sig, bkg, unc):
    return ROOT.RooStats.NumberCountingUtils.BinomialExpZ(sig, bkg, unc)


def _debug_memory_usage():
    if LOG.isEnabledFor(logging.DEBUG):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        LOG.debug('memory usage: %.3f MB', rss / 1000.0)


def _info_sr(srdict, srname, group):
    no_z = 'expected_Z' not in srdict[srname][group]
    fmt = '    %s: yield=%.3e, uncert=%.3e, eff=%.3e'
    fmt_z = '%s' if no_z else ' Z=%.3f'
    LOG.info(
        fmt + fmt_z,
        group,
        srdict[srname][group]['yield'],
        srdict[srname][group]['yield_uncert'],
        srdict[srname][group]['efficiency'],
        '' if no_z else srdict[srname][group]['expected_Z']
    )


def _info_count_group(cnt_dict, group):
    LOG.info(
        '  %s: unweighted=%d, weighted=%.3e',
        group,
        cnt_dict[group]['unweighted'],
        cnt_dict[group]['weighted'],
    )


def _count_initial(ddict, lumi):
    LOG.info('Counting initial statistics')
    cnt_dict = collections.OrderedDict()
    for group, dset in ddict.iteritems():
        cnt_dict[group] = {
            'unweighted': dset.shape[0],
            'weighted': np.sum(dset['M_weight']) * lumi
        }
        _info_count_group(cnt_dict, group)

    bkg_tot_u = sum(
        [cnt_dict[g]['unweighted']
         for g in cnt_dict if not g.startswith('Gtt')]
    )
    bkg_tot_w = sum(
        [cnt_dict[g]['weighted']
         for g in cnt_dict if not g.startswith('Gtt')]
    )
    cnt_dict['background'] = {
        'unweighted': bkg_tot_u,
        'weighted': bkg_tot_w
    }
    _info_count_group(cnt_dict, 'background')

    sig_tot_u = sum(
        [cnt_dict[g]['unweighted']
         for g in cnt_dict if g.startswith('Gtt')]
    )
    sig_tot_w = sum(
        [cnt_dict[g]['weighted']
         for g in cnt_dict if g.startswith('Gtt')]
    )
    cnt_dict['signal'] = {
        'unweighted': sig_tot_u,
        'weighted': sig_tot_w
    }
    _info_count_group(cnt_dict, 'signal')

    return cnt_dict


def _count_signal_regions(ddict, cnt_dict, lumi, uncert):
    LOG.info('Counting statistics in signal regions')

    srdict = collections.defaultdict(
        functools.partial(collections.defaultdict, dict)
    )

    for srname, srfunc in signal_regions.SR_dict.iteritems():
        LOG.info('  %s', srname)
        for group in [g for g in ddict if not g.startswith('Gtt')]:
            ysr, dysr = srfunc(ddict[group])
            srdict[srname][group] = {
                'yield': ysr * lumi,
                'yield_uncert': dysr * lumi,
                'efficiency': ysr * lumi / cnt_dict[group]['weighted']
            }
            _info_sr(srdict, srname, group)
        byield = sum([d['yield'] for d in srdict[srname].itervalues()])
        byield_uncert = np.sqrt(
            sum([d['yield_uncert'] ** 2 for d in srdict[srname].itervalues()])
        )
        srdict[srname]['background'] = {
            'yield': byield,
            'yield_uncert': byield_uncert,
            'efficiency': byield / cnt_dict['background']['weighted']
        }
        _info_sr(srdict, srname, 'background')

        for group in [g for g in ddict if g.startswith('Gtt')]:
            ysr, dysr = srfunc(ddict[group])
            srdict[srname][group] = {
                'yield': ysr * lumi,
                'yield_uncert': dysr * lumi,
                'efficiency': ysr * lumi / cnt_dict[group]['weighted'],
                'expected_Z': _expected_z(
                    ysr * lumi,
                    srdict[srname]['background']['yield'],
                    uncert
                )
            }
            _info_sr(srdict, srname, group)

    return srdict


def _load_data(path, ttbar):
    LOG.info('Loading dataset')
    ddict = collections.OrderedDict()
    h5dset = h5.File(path, 'r')
    for group in [g for g in h5dset.keys() if g != 'Gtt']:
        if group == 'ttbar' and ttbar != 'ttbar':
            continue
        if group == 'MGPy8EG_ttbar' and ttbar != 'MGPy8EG_ttbar':
            continue
        if group == 'PhHppEG_ttbar' and ttbar != 'PhHppEG_ttbar':
            continue
        LOG.info('  %s', group)
        ddict[group] = h5dset[group][:]
        _debug_memory_usage()
    LOG.info('Fetching all Gtt datasets')
    gtt_dset = h5dset['Gtt'][:]
    _debug_memory_usage()
    masses = np.unique(gtt_dset[['I_m_gluino', 'I_m_lsp']])
    for m_g, m_l in masses:
        key = 'Gtt_{}_{}'.format(int(m_g), int(m_l))
        LOG.info('  %s', key)
        sel = np.where(
            (gtt_dset['I_m_gluino'] == m_g) & (gtt_dset['I_m_lsp'] == m_l)
        )[0]
        ddict[key] = gtt_dset[sel]
        _debug_memory_usage()

    return ddict


def _to_pkl(path, cnt_dict, srdict):
    if path is not None:
        if not path.endswith('.pkl'):
            path += '.pkl'
        LOG.info('Saving dicts to %s', path)
        with open(path, 'w') as pkl:
            LOG.debug(
                '%s %s %s',
                type(cnt_dict),
                type(srdict),
                type((cnt_dict, srdict))
            )
            cPickle.dump((cnt_dict, srdict), pkl)


def _from_pkl(path):
    LOG.info('Loading dicts from %s', path)
    with open(path, 'r') as pkl:
        return cPickle.load(pkl)


def _add_count_table(doc, cnt_dict):
    with doc.create(latex.Section('Initial statistics, background')):
        table = latex.LongTable('l c c', pos='h!')
        table.add_row(
            latex.utils.bold('Sample'),
            'Unweighted count',
            'Weighted count'
        )
        table.add_hline()
        bgroups = [
            g for g in cnt_dict if not g.startswith('Gtt') and
            g not in ['signal', 'background']
        ]
        for grp in bgroups:
            table.add_row(
                latex.utils.bold(grp),
                cnt_dict[grp]['unweighted'],
                '{:.3e}'.format(cnt_dict[grp]['weighted'])
            )
        table.add_hline()
        table.add_row(
            latex.utils.bold('Total'),
            cnt_dict['background']['unweighted'],
            '{:.3e}'.format(cnt_dict['background']['weighted'])
        )
        doc.append(table)

    with doc.create(latex.Section('Initial statistics, signal')):
        table = latex.LongTable('l c c', pos='h!')
        table.add_row(
            latex.utils.bold('Sample'),
            'Unweighted count',
            'Weighted count'
        )
        table.add_hline()
        for grp in sorted([g for g in cnt_dict if g.startswith('Gtt')], _gsr):
            table.add_row(
                latex.utils.bold(grp),
                cnt_dict[grp]['unweighted'],
                '{:.3e}'.format(cnt_dict[grp]['weighted'])
            )
        table.add_hline()
        table.add_row(
            latex.utils.bold('Total'),
            cnt_dict['signal']['unweighted'],
            '{:.3e}'.format(cnt_dict['signal']['weighted'])
        )
    doc.append(table)


def _gsr(gs1, gs2):
    def _key(gsi):
        _, f_1, f_2 = gsi.split('_')
        return int(f_1), int(f_2)

    return cmp(_key(gs1), _key(gs2))


def _srsort(sr1, sr2):
    def _key2(sri):
        return {'B': 0, 'M': 1, 'C': 2}[sri[-1]]

    def _key1(sri):
        return 0 if '_0l_' in sri else 1

    return cmp((_key1(sr1), _key2(sr1)), (_key1(sr2), _key2(sr2)))


def _add_sr_table(doc, srdict):
    # pylint: disable=invalid-name
    for sr in sorted(srdict.iterkeys(), cmp=_srsort):
        sec = 'Signal region {}, background'.format(sr)
        with doc.create(latex.Section(sec)):
            table = latex.LongTable('l c c c', pos='h!')
            table.add_row(
                latex.utils.bold('sample'),
                'yield',
                'rel. uncert.',
                'efficiency',
            )
            table.add_hline()
            bgroups = [
                g for g in srdict[sr] if not g.startswith('Gtt') and
                g not in ['signal', 'background']
            ]
            for grp in bgroups:
                y = srdict[sr][grp]['yield']
                dy = srdict[sr][grp]['yield_uncert']
                eff = srdict[sr][grp]['efficiency']
                table.add_row(
                    latex.utils.bold(grp),
                    '{:.3e}'.format(y),
                    '{:.3e}'.format(dy / y),
                    '{:.3e}'.format(eff),
                )
            table.add_hline()
            y = srdict[sr]['background']['yield']
            dy = srdict[sr]['background']['yield_uncert']
            eff = srdict[sr]['background']['efficiency']
            table.add_row(
                latex.utils.bold('Total'),
                '{:.3e}'.format(y),
                '{:.3e}'.format(dy / y),
                '{:.3e}'.format(eff),
            )
            doc.append(table)

        sec = 'Signal region {}, signal'.format(sr)
        with doc.create(latex.Section(sec)):
            table = latex.LongTable('l c c c c', pos='h!')
            table.add_row(
                latex.utils.bold('sample'),
                'yield',
                'rel. uncert.',
                'efficiency',
                'expected Z',
            )
            table.add_hline()
            srgrp = [g for g in srdict[sr] if g.startswith('Gtt')]
            for grp in sorted(srgrp, cmp=_gsr):
                y = srdict[sr][grp]['yield']
                dy = srdict[sr][grp]['yield_uncert']
                eff = srdict[sr][grp]['efficiency']
                z = srdict[sr][grp]['expected_Z']
                table.add_row(
                    latex.utils.bold(grp),
                    '{:.3e}'.format(y),
                    '{:.3e}'.format(dy / y),
                    '{:.3e}'.format(eff),
                    '{:.3f}'.format(z)
                )
            doc.append(table)


def _to_pdf(path, cnt_dict, srdict, lumi):

    LOG.debug(type(srdict))

    if path is None:
        return

    LOG.info('Creating pdf report')

    name = path.replace('.pdf', '')
    doc = latex.Document(name)
    doc.preamble.append(
        latex.Command(
            'title',
            latex.NoEscape('Statistics report @ %.1f fb$^{-1}$' % lumi)
        )
    )
    doc.preamble.append(latex.Command('date', latex.NoEscape(r'\today')))
    doc.append(latex.NoEscape(r'\maketitle'))

    _add_count_table(doc, cnt_dict)
    _add_sr_table(doc, srdict)

    doc.generate_pdf()
    LOG.info('Generated %s', name + '.pdf')


###############################################################################
# Main


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('input')
    args.add_argument(
        '--ttbar',
        choices=['ttbar', 'PhHppEG_ttbar', 'MGPy8EG_ttbar'],
        default='ttbar'
    )
    args.add_argument('--lumi', type=float, default=1.0)
    args.add_argument('--uncert', type=float, default=0.3)
    args.add_argument('--to-pkl')
    args.add_argument('--from-pkl')
    args.add_argument('--to-pdf')
    return args.parse_args()


def _main():
    args = _get_args()
    LOG.info('input: %s', args.input)
    LOG.info('luminosity: %f', args.lumi)
    LOG.info('ttbar: %s', args.ttbar)

    if args.from_pkl is not None:
        cnt_dict, srdict = _from_pkl(args.from_pkl)
    else:
        ddict = _load_data(args.input, args.ttbar)
        _debug_memory_usage()
        cnt_dict = _count_initial(ddict, args.lumi)
        srdict = _count_signal_regions(ddict, cnt_dict, args.lumi, args.uncert)

    _to_pkl(args.to_pkl, cnt_dict, srdict)
    _to_pdf(args.to_pdf, cnt_dict, srdict, args.lumi)

if __name__ == '__main__':
    utils.main(_main, 'stats.py')
