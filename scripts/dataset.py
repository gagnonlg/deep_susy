import collections
import glob
import logging

import ROOT

import utils

log = logging.getLogger(__name__)


def lookup_by_dsid(datadir, dsid, treename):
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


def lookup(datadir, treename):
    cfg = __read_config()
    ddict = collections.defaultdict(list)
    data = collections.namedtuple(
        'Dataset',
        ['dsid', 'tree'],
    )
    for (key, lst) in cfg.iteritems():
        for dsid in lst:
            ddict[key].append(
                data(
                    dsid=dsid,
                    tree=lookup_by_dsid(datadir, dsid, treename)
                )
            )
    return ddict
