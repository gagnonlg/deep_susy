""" Functions to preprocess the dataset """
import logging

import numpy as np

from deep_susy import custom_layers


def _norm(obj):
    order = len(obj)
    if order == 1:
        return np.abs(obj)
    elif order == 2 or order == 3:
        return np.sqrt(np.dot(obj, obj))
    elif order == 4:
        return np.sqrt(obj[3]*obj[3] - _norm(obj[:3]))


def _fill_vec_scale_factor(vlen, dset, start_indices, scales):
    logging.debug('start_indices: %s', start_indices)
    avg = np.empty((len(start_indices), vlen))
    for i, idx in enumerate(start_indices):
        avg[i] = np.mean(dset[:, idx:idx+vlen], axis=0)
    avg = np.mean(avg, axis=0)
    fact = 1.0 / _norm(avg)
    for idx in start_indices:
        scales[idx:idx+vlen] = fact


def _fill_scalar_scale_factor(dset, indices, scales, offsets):
    logging.debug('indices: %s', indices)
    for i in indices:
        maxv = np.max(dset[:, i])
        minv = np.min(dset[:, i])
        midrange = 0.5 * (maxv + minv)
        vrange = maxv - minv
        scales[i] = 2.0 / vrange
        offsets[i] = - midrange * scales[i]


def normalization(dset, xdtype):
    """Normalize the dataset in a 4-vector-aware scheme

    Assumed dataset structure:

    <small-R jets: px, py, pz, e, isb>...
    <large-R jets: px, py, pz, e>...
    <leptons: px, py, pz, e>...
    <met: px, py>
    <masses: m_gluino, m_lsp>

    The 4-vectors are normalized to have unit norm on average. This
    normalization is done separately for small-R jets, large-R jets
    and leptons.

    The the isb binary variable and the masses are rescaled
    to then -1, 1 range.

    The MET 2-vector is rescaled to unit norm.
    """

    scale = np.ones(dset.shape[1])
    offset = np.zeros(dset.shape[1])

    if xdtype is None:
        logging.warning("No dtype provided, can't compute normalization")
        return custom_layers.ScaleOffset(
            scale=scale.astype('float32'),
            offset=offset.astype('float32')
        )

    logging.debug(xdtype)

    hdr = [(i, n) for (i, (n, _)) in enumerate(xdtype.descr)]
    logging.debug('hdr: %s', hdr)

    if any('small_R_jets_pt' in name for (_, name) in hdr):
        raise RuntimeError(
            '4-vector aware normalization scheme only defined '
            'for px,py,pz,e parametrization!'
        )

    scale = np.ones(dset.shape[1])
    offset = np.zeros(dset.shape[1])

    logging.debug('small_R_jets_px')
    _fill_vec_scale_factor(
        4,
        dset,
        start_indices=[i for (i, name) in hdr if 'small_R_jets_px' in name],
        scales=scale
    )

    logging.debug('_isb_' )
    _fill_scalar_scale_factor(
        dset,
        indices=[i for (i, name) in hdr if '_isb_' in name],
        scales=scale,
        offsets=offset
    )

    logging.debug('large_R_jets_px')
    _fill_vec_scale_factor(
        4,
        dset,
        start_indices=[i for (i, name) in hdr if 'large_R_jets_px' in name],
        scales=scale
    )

    logging.debug('leptons_px')
    _fill_vec_scale_factor(
        4,
        dset,
        start_indices=[i for (i, name) in hdr if 'leptons_px' in name],
        scales=scale
    )

    logging.debug('_met_px')
    _fill_vec_scale_factor(
        2,
        dset,
        start_indices=[i for (i, name) in hdr if '_met_px' in name],
        scales=scale
    )

    logging.debug('I_m_')
    _fill_scalar_scale_factor(
        dset,
        indices=[i for (i, name) in hdr if name.startswith('I_m_')],
        scales=scale,
        offsets=offset
    )

    logging.debug('scale: %s', scale)
    logging.debug('offset: %s', offset)

    return custom_layers.ScaleOffset(
        scale=scale.astype('float32'),
        offset=offset.astype('float32')
    )


def standardize(dset, dtype):
    """ Standardization from 1402.4735 """
    branches = [n for n, _ in dset.dtype.descr]
    scales = np.ones(len(branches))
    offsets = np.zeros(len(branches))
    means = np.mean(dset, axis=0)
    stds = np.std(dset, ddof=1, axis=0)
    scales = 1.0 / stds
    offsets = - means / stds

    for i, pos in enumerate(np.all(np.greater_equal(dset, 0), axis=0)):
        if pos:
            offsets[i] += 1

    return custom_layers.ScaleOffset(
        scales.astype('float32'),
        offsets.astype('float32')
    )
