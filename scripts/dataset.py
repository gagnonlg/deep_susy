import collections
import glob
import logging

import ROOT

import utils

log = logging.getLogger(__name__)


def lookup_by_dsid(datadir, dsid, treename, xsec=False):
    log.debug('datadir: %s', datadir)
    log.debug('dsid: %s', dsid)
    log.debug('treename: %s', treename)
    # first look for subdirectory
    dset = glob.glob('{}/*.{}.*.root/*.root*'.format(datadir, dsid))
    if len(dset) == 0:
        # look for single file
        dset = glob.glob('{}/{}.*.root'.format(datadir, dsid))
        if len(dset) == 0:
            raise RuntimeError('dsid %s not found', dsid)
    chain = ROOT.TChain(treename)
    for dpath in dset:
        chain.Add(dpath)
    if xsec:
        return chain, __get_xsec(dset)
    else:
        return chain


def __read_config():
    path = utils.top_directory() + '/datasets.config'
    key = None
    ddict = collections.defaultdict(list)
    with open(path, 'r') as dfile:
        for line in dfile:
            if line.startswith('#'):
                key = line.strip('# \n')
                log.debug(key)
            else:
                ddict[key].append(line.split(' ')[0])
                log.debug('     %s', ddict[key][-1])
    return ddict


def lookup(datadir, treename, xsec=False):
    cfg = __read_config()
    ddict = collections.defaultdict(list)
    data = collections.namedtuple(
        'Dataset',
        ['dsid', 'tree', 'xsec']
    )
    for (key, lst) in cfg.iteritems():
        for dsid in lst:
            if xsec:
                tree, xsecv = lookup_by_dsid(datadir, dsid, treename, xsec=True)
            else:
                tree = lookup_by_dsid(datadir, dsid, treename)
                xsecv = -1
            ddict[key].append(
                data(
                    dsid=dsid,
                    tree=tree,
                    xsec=xsecv
                )
            )
    return ddict


def __get_xsec(paths):
    weight = 0
    for path in paths:
        tf = ROOT.TFile(path, 'READ')
        hxsec = tf.Get('cross_section')
        xsec = hxsec.GetBinContent(1) / hxsec.GetEntries()
        hcut = tf.Get('cut_flow')
        weight += hcut.GetBinContent(2)
    return 1000.0 * xsec / weight
