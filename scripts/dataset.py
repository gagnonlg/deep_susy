import collections
import glob
import logging

import h5py as h5
import numpy as np
import ROOT
import root_numpy

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


def __load(datalist):
    nrow = np.sum([c.tree.GetEntries() for c in datalist])

    # get all the data into an array
    for i, data in enumerate(datalist):
        log.debug('__load dsid=%s', data.dsid)
        subarray = root_numpy.tree2array(data.tree)
        if i == 0:
            array = np.empty(nrow, dtype=subarray.dtype)
            i0 = 0
        else:
            i0 = i1
        i1 = i0 + subarray.shape[0]
        array[i0:i1] = subarray

    # Add the gluino and lsp masses for the Gtt samples
    try:
        mg, ml = gtt.get_masses(int(data.dsid))
        array[:, 'I_m_gluino'] = mg
        array[:, 'I_m_lsp'] = ml
    except KeyError:
        pass

    # Shuffle the group
    np.random.shuffle(array)

    # Split input from metadata
    inputv = array[:, __get_slice(array, lambda n: n.startswith('I_'))]
    metadv = array[:, __get_slice(array, lambda n: n.startswith('M_'))]

    # Destructure and downcast input array
    inputv = structured.view(np.float64).reshape(structured.shape + (-1,)).as_type(np.float32)

    return inputv, metadv

def __get_slice(sarray, selector):
    names = __get_header(sarray)

    indices = [i for i, name in enumerate(names) if selector(name)]

    LOGGER.debug('indices: %s', str(indices))

    ilow = min(indices)
    iup = max(indices) + 1

    if not indices == range(ilow, iup):
        raise RuntimeError('selector yields non-contiguous indices')

    if iup - ilow == 1:
        return ilow
    else:
        return slice(ilow, iup)



def create_master(datadir, output):

    ddict = lookup(datadir, 'NNinput')
    with h5.File(utils.unique_path(output), 'x') as outf:
        for group in ddict:
            log.info('Adding group: %s', group)
            outf.create_dataset(
                name=group,
                data=__load(ddict[group]),
                compression='gzip',
                chunks=True,
                shuffle=True,
                track_times=True
            )

