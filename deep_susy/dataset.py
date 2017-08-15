""" Dataset-related code """
import collections
import glob
import itertools
import logging
import re
import warnings

import h5py as h5
import numpy as np

try:
    import ROOT
    import root_numpy
except ImportError:
    warnings.warn('ROOT not found')

from deep_susy import gtt, utils

LOG = logging.getLogger(__name__)

TARGETS = (
    'Gtt',
    'ttbar',
    'singletop',
    'topEW',
    'WZjets',
)


def _get_h5_keys(h5dset, fold, dsetname):
    keys = []

    def _select(key):
        if fold in key and dsetname in key:
            keys.append(key)
    h5dset.visit(_select)
    return keys


def unpack(splith5, fold, dsetname, destructure=True):
    """ Unpack the dset """

    if fold not in ['training', 'validation', 'test']:
        raise RuntimeError(
            "fold == %s, not in ['training', 'validation', 'test']",
            fold
        )

    if fold not in splith5.keys():
        raise RuntimeError("fold == %s, not in h5 file")

    if dsetname not in ['input', 'target', 'metadata']:
        raise RuntimeError(
            "fold == %s, not in ['input', 'target', 'metadata']"
        )

    # compute the size
    keys = _get_h5_keys(splith5, fold, dsetname)
    dtype = ncols = None
    for k in keys:
        if splith5[k].size > 0:
            dtype = splith5[k].dtype
            if len(splith5[k].shape) == 2:
                ncols = splith5[k].shape[1]
    nrows = np.sum([splith5[k].shape[0] for k in keys])

    # initialize array
    if ncols is not None:
        out_dset = np.empty(shape=(nrows, ncols))
    else:
        out_dset = np.empty(shape=nrows, dtype=dtype)

    # fetch all groups
    i_0 = i_1 = 0
    for key in keys:
        dset = splith5[key]
        i_1 += dset.shape[0]
        if i_1 > i_0:
            out_dset[i_0:i_1] = np.copy(dset)
        i_0 = i_1

    # destructure
    if destructure and ncols is None:
        return destructure_array(out_dset)
    elif ncols is not None:
        return out_dset.astype(np.float32)

    return out_dset


def destructure_array(array):
    return array.view(np.float64).reshape(
        array.shape + (-1,)
    ).astype(np.float32)


def create_split(inputp,
                 outputp,
                 default_fractions=(0.5, 0.25, 0.25),
                 custom_fractions=None):
    """ Split master dataset into training, validation & test folds """

    inh5 = h5.File(inputp, 'r')
    outp = utils.unique_path(outputp)
    fractions = _get_fractions(inh5, default_fractions, custom_fractions)
    indices = _generate_indices(inh5, fractions)

    LOG.info('create_split')
    LOG.info('  input path: %s', inputp)
    LOG.info('  output path: %s', outp)
    LOG.info('  default_fractions: %s', default_fractions)
    LOG.info('  custom_fractions: %s', custom_fractions)

    with h5.File(outp, 'x') as outh5:
        for i, split in enumerate(['training', 'validation', 'test']):
            LOG.info('Creating split: %s', split)
            grp = outh5.create_group(split)
            _split(i, grp, inh5, indices)
    return outp


def create_master(datadir, output):
    """ Create master h5 dataset """
    ddict = lookup(datadir, 'NNinput')
    path = utils.unique_path(output)
    with h5.File(path, 'x') as outf:
        signal_grp = outf.create_group('signal')
        bkgnd_grp = outf.create_group('background')
        for group in ddict:
            if group == 'Diboson':
                LOG.warning('Ignoring diboson')
                continue
            grp = signal_grp if group.startswith('Gtt_') else bkgnd_grp
            sub_grp = grp.create_group(group)
            LOG.info(sub_grp.name)
            data = __load(ddict[group])
            cols = [n for n, _ in data.dtype.descr]
            sub_grp.create_dataset(
                name='input',
                data=data[[c for c in cols if c.startswith('I_')]],
                compression='gzip',
                chunks=True,
            )
            sub_grp.create_dataset(
                name='metadata',
                data=data[[c for c in cols if c.startswith('M_')]],
                compression='gzip',
                chunks=True,
            )
            sub_grp.create_dataset(
                name='target',
                data=np.tile(_target(group), (data.shape[0], 1)),
                compression='gzip',
                chunks=True
            )
    return path


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


def _target(grp):
    target = np.zeros(5)
    if grp.startswith('Gtt'):
        target[0] = 1
    elif 'ttbar' in grp:
        target[1] = 1
    elif grp == 'singletop':
        target[2] = 1
    elif grp == 'topEW':
        target[3] = 1
    elif grp == 'WZjets':
        target[4] = 1
    return target


def _get_fractions(inh5, default_fractions, custom_fractions):
    fractions = collections.OrderedDict([
        (key, default_fractions) for key in
        itertools.chain(inh5['signal'].keys(), inh5['background'].keys())
    ])
    if custom_fractions is not None:
        fractions.update(custom_fractions)
    return fractions


def _generate_indices(inh5, fractions):
    indices = collections.defaultdict(list)
    signal_keys = ['signal/' + k for k in inh5['signal'].keys()]
    bkgnd_keys = ['background/' + k for k in inh5['background'].keys()]
    for key in itertools.chain(signal_keys, bkgnd_keys):
        short_key = key.split('/')[-1]
        dset = inh5[key + '/input']
        idx = np.arange(0, dset.shape[0])
        np.random.shuffle(idx)
        endpoints = np.cumsum([
            int(round(f * dset.shape[0])) for f
            in fractions[short_key]
        ])
        i_0 = 0
        for i_1 in endpoints:
            indices[short_key].append(idx[i_0:i_1])
            i_0 = i_1
    return indices


def _update_mass(masses_dist, input_data, meta_data, key):
    m_g = input_data[0]['I_m_gluino']
    m_l = input_data[0]['I_m_lsp']

    if m_g == 0.0 or m_l == 0.0:
        LOG.warning('NULL masses for %s', key)
        match = re.search('Gtt_(.*)_5000_(.*)', key)
        m_g = float(match.group(1))
        m_l = float(match.group(2))
        input_data['I_m_gluino'] = m_g
        input_data['I_m_lsp'] = m_l
        LOG.warning('recovered mg=%f, ml=%f', m_g, m_l)

    masses_dist['probs'].append(np.sum(meta_data['M_weight']))
    masses_dist['mgs'].append(m_g)
    masses_dist['mls'].append(m_l)
    LOG.debug(masses_dist['mgs'][-1])
    LOG.debug(masses_dist['mls'][-1])


def _normalize_mass(masses_dist):
    masses_dist.default_factory = None
    tot = np.sum(masses_dist['probs'])
    masses_dist['probs'] = np.array(
        [n / float(tot) for n in masses_dist['probs']]
    )
    masses_dist['mgs'] = np.array(masses_dist['mgs'])
    masses_dist['mls'] = np.array(masses_dist['mls'])

    np.testing.assert_almost_equal(
        1.0,
        np.sum(masses_dist['probs'])
    )


def _sample_mass(masses_dist, input_data):
    i_masses = np.random.choice(
        masses_dist['probs'].shape[0],
        p=masses_dist['probs'],
        size=input_data.shape[0]
    ).astype(int)
    input_data['I_m_gluino'] = masses_dist['mgs'][i_masses]
    input_data['I_m_lsp'] = masses_dist['mls'][i_masses]
    LOG.debug(input_data['I_m_gluino'])
    LOG.debug(input_data['I_m_lsp'])


def _reweight(indset, meta_data):
    meta_data['M_weight'] *= (
        np.sum(indset['metadata']['M_weight']) /
        np.sum(meta_data['M_weight'])
    )


def _split_one(inh5, key, outgrp, indices, masses_dist):

    if indices.size == 0:
        LOG.warning('Requested split is empty')
        input_data = np.array([])
        meta_data = np.array([])
        target_data = np.array([])
    else:
        indset = inh5[key]
        input_data = indset['input'][:][indices]
        meta_data = indset['metadata'][:][indices]
        target_data = indset['target'][:][indices]

        _reweight(indset, meta_data)

        if 'background' in key:
            _sample_mass(masses_dist, input_data)
        else:
            _update_mass(masses_dist, input_data, meta_data, key)

    outgrp.create_dataset(key + '/input', data=input_data)
    outgrp.create_dataset(key + '/metadata', data=meta_data)
    outgrp.create_dataset(key + '/target', data=target_data)


def _split(i, grp, inh5, indices):
    masses_dist = collections.defaultdict(list)

    for key in ['signal/' + k for k in inh5['signal'].keys()]:
        LOG.info(key)
        _split_one(
            inh5=inh5,
            key=key,
            outgrp=grp,
            indices=indices[key.split('/')[-1]][i],
            masses_dist=masses_dist
        )

    _normalize_mass(masses_dist)

    for key in ['background/' + k for k in inh5['background'].keys()]:
        LOG.info(key)
        _split_one(
            inh5=inh5,
            key=key,
            outgrp=grp,
            indices=indices[key.split('/')[-1]][i],
            masses_dist=masses_dist,
        )


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

    indices = utils.range_sequence([c.tree.GetEntries() for c in datalist])

    # get all the data into an array
    for i, (data, (i_0, i_1)) in enumerate(zip(datalist, indices)):
        LOG.debug('__load dsid=%s', data.dsid)
        subarray = root_numpy.tree2array(data.tree)
        if i == 0:
            array = np.empty(indices[-1][1], dtype=subarray.dtype)

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
