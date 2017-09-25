""" Functions to evaluate a model's performance """
import logging

import numpy as np
import ROOT


def compute_threshold(evaluated, sigkey, metadata):
    """ Compute decision threshold

    Compute the decision theshold on the first output probability bin,
    P_SUSY, such that the statistical uncertainty on the ttbar sample
    doesn't exceed 30%.

    Arguments:
      -- evaluated: H5 dataset containing the output probabilities for
         all mass points
      -- sigkey: Key indexing the wanted mass point
      -- metadata: metatada, to get the weights
    Returns: The decision threshold
    """

    ttbar = None
    for k in _bkg_keys(evaluated, sigkey):
        if 'ttbar' in k:
            ttbar = k
            break
    logging.debug('ttbar: %s', ttbar)

    weights = metadata['background/' + ttbar + '/metadata'].value['M_weight']
    scores = evaluated[sigkey + '/background/' + ttbar].value[:, 0]

    isort = np.argsort(scores)[::-1]
    scores = scores[isort]
    weights = weights[isort]

    wsum = np.cumsum(weights)
    wsum2 = np.sqrt(np.cumsum(weights * weights))
    funcert = wsum2 / wsum

    imin = np.argmin(np.abs(funcert - 0.3))

    return scores[imin]


def compute_yield(scores, weights, threshold):
    """ Compute yield for a luminosity of 1 fb^-1
    Arguments:
      -- scores: the model's output, event-by-event
      -- weights: events-wise 1 fb^-1 weight
      -- threshold: the decision threshold
    Returns: The 1 fb^-1 yield
    """
    logging.debug('scores: %s, weights: %s', scores.shape, weights.shape)
    return np.sum(weights[np.where(scores[:, 0] > threshold)])


def compute_significance_grid(evaluated, data, lumi, uncert):
    """ Compute significance across the whole mass grid

    Arguments:
      -- evaluated: H5 dataset containing the output probabilities for
         all mass points
      -- data: H5 containing the input data, to get the weights
      -- lumi: target luminosity
      -- uncert: systematic uncertainty used in significance computation
    Returns: a structured array with keys mg, ml and z.
    """

    results = np.zeros(
        len(evaluated.keys()),
        dtype=[
            ('mg', 'i4'),
            ('ml', 'i4'),
            ('z', 'f4'),
            ('s', 'f4'),
            ('b', 'f4')
        ]
    )

    for i, sigkey in enumerate(evaluated.keys()):
        thr = compute_threshold(evaluated, sigkey, data)
        s_yield = compute_yield(
            scores=evaluated[sigkey + '/signal/' + sigkey].value,
            weights=data['signal/' + sigkey + '/metadata'].value['M_weight'],
            threshold=thr
        )

        b_yield = 0
        for bkgkey in _bkg_keys(evaluated, sigkey):
            logging.debug('  %s', bkgkey)
            b_yield += compute_yield(
                scores=evaluated[sigkey + '/background/' + bkgkey].value,
                weights=data['background/' + bkgkey + '/metadata'].value[
                    'M_weight'
                ],
                threshold=thr
            )

        expz = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(
            s_yield * lumi,
            b_yield * lumi,
            uncert
        )

        fields = sigkey.split('_')
        results[i] = (
            int(fields[1]),
            int(fields[3]),
            expz,
            s_yield * lumi,
            b_yield * lumi
        )
        logging.info('%s: %f', sigkey, expz)

    return results


def compute_n_excluded(grid):
    """ Compute the number of excluded points at 95% CL """
    return np.count_nonzero(grid[np.where(grid['z'] >= 1.64)])


def _bkg_keys(dfile, sigkey):

    for key in dfile[sigkey + '/background']:
        if dfile[sigkey + '/background/' + key].shape[0] > 0:
            yield key
