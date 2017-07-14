""" Dataset-related code """
import collections
import glob
import logging

import h5py as h5
import numpy as np
import ROOT
import root_numpy

import gtt
import utils

LOG = logging.getLogger(__name__)

TARGETS = (
    'Gtt',
    'ttbar',
    'singletop',
    'topEW',
    'WZjets',
)


def create_master(datadir, output):
    """ Create master h5 dataset """
    ddict = lookup(datadir, 'NNinput')
    with h5.File(utils.unique_path(output), 'x') as outf:
        for group in ddict:
            LOG.info('Adding group: %s', group)
            outf.create_dataset(
                name=group,
                data=__load(ddict[group]),
                compression='gzip',
                chunks=True,
            )


def lookup(datadir, treename, xsec=False):
    """ Lookup a many-file dataset """
    cfg = __read_config()
    ddict = collections.defaultdict(list)
    data = collections.namedtuple(
        'Dataset',
        ['dsid', 'tree', 'xsec', 'descr']
    )
    for (key, lst) in cfg.iteritems():
        for dsid, descr in lst:
            if xsec:
                tree, xsecv = lookup_by_dsid(
                    datadir,
                    dsid,
                    treename,
                    xsec=True
                )
            else:
                tree = lookup_by_dsid(datadir, dsid, treename)
                xsecv = -1
            ddict[key].append(
                data(
                    dsid=dsid,
                    tree=tree,
                    xsec=xsecv,
                    descr=descr
                )
            )
    return ddict


def lookup_by_dsid(datadir, dsid, treename, xsec=False):
    """ Lookup a many-file dataset with common dsid """
    LOG.debug('datadir: %s', datadir)
    LOG.debug('dsid: %s', dsid)
    LOG.debug('treename: %s', treename)
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


def __get_xsec(paths):
    weight = 0
    for path in paths:
        tfile = ROOT.TFile(path, 'READ')
        hxsec = tfile.Get('cross_section')
        xsec = hxsec.GetBinContent(1) / hxsec.GetEntries()
        hcut = tfile.Get('cut_flow')
        weight += hcut.GetBinContent(2)
    return 1000.0 * xsec / weight


def __load(datalist):
    nrow = np.sum([c.tree.GetEntries() for c in datalist])

    # get all the data into an array
    for i, data in enumerate(datalist):
        LOG.debug('__load dsid=%s', data.dsid)
        subarray = root_numpy.tree2array(data.tree)
        i_0 = i_1 = 0
        if i == 0:
            array = np.empty(nrow, dtype=subarray.dtype)
        else:
            i_0 = i_1
        i_1 = i_0 + subarray.shape[0]

        # Add the gluino and lsp masses for the Gtt samples
        try:
            m_g, m_l = gtt.get_masses(int(data.dsid))
            subarray['I_m_gluino'] = m_g
            subarray['I_m_lsp'] = m_l
        except KeyError:
            pass

        array[i_0:i_1] = np.copy(subarray)

    return array


def __read_config():
    path = utils.top_directory() + '/datasets.config'
    key = None
    ddict = collections.defaultdict(list)
    with open(path, 'r') as dfile:
        for line in dfile:
            if line.startswith('#'):
                key = line.strip('# \n')
                LOG.debug(key)
            else:
                fields = line.strip().split(' ')
                ddict[key].append((fields[0], fields[1]))
                LOG.debug('     %s', ddict[key][-1][0])
    return ddict
