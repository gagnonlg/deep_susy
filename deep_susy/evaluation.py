""" Functions to evaluate a model's performance """
import logging

import numpy as np
import scipy.special
import sklearn.metrics

def significance(signalExp, backgroundExp, relativeBkgUncert):
    """ Numpy/Scipy port of the RooStats function `BinomialExpZ'

    See: https://root.cern.ch/doc/master/NumberCountingUtils_8cxx_source.html
    """
    mainInf = signalExp + backgroundExp
    tau = 1.0 / backgroundExp / (relativeBkgUncert * relativeBkgUncert)
    auxiliaryInf = backgroundExp * tau
    P_Bi = scipy.special.betainc(mainInf, auxiliaryInf + 1, 1.0 / (1.0 + tau))
    return - scipy.special.ndtri(P_Bi)


def compute_threshold_ttbar(evaluated, sigkey, metadata):
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

    funcert, scores = _compute_ttbar_funcert(evaluated, sigkey, metadata, return_scores=True)
    return np.max(scores[np.where(funcert <= 0.3)])

def _compute_ttbar_funcert(evaluated, sigkey, metadata, thr=None, return_scores=False):
    ttbar = _ttbar_key(evaluated, sigkey)
    logging.debug('ttbar: %s', ttbar)

    weights = metadata['background/' + ttbar + '/metadata'].value['M_weight']
    scores = evaluated[sigkey + '/background/' + ttbar].value[:, 0]

    if thr is None:
        isort = np.argsort(scores)[::-1]
        scores = scores[isort]
        weights = weights[isort]
        wsum = np.cumsum(weights)
        wsum2 = np.sqrt(np.cumsum(weights * weights))
    else:
        wsel = weights[np.where(scores >= thr)]
        wsum = np.sum(wsel)
        wsum2 = np.sqrt(np.sum(wsel * wsel))

    funcert = wsum2 / wsum
    if return_scores:
        return funcert, scores
    return funcert

def compute_threshold(evaluated, sigkey, data, lumi, uncert, return_effs=False):
    s_scores, s_weights = _get_scores_and_weights(
        evaluated[sigkey + '/signal'],
        data['signal/'],
        [sigkey]
    )

    ttbar = _ttbar_key(evaluated, sigkey)
    tt_scores, tt_weights = _get_scores_and_weights(
        evaluated[sigkey + '/background/'],
        data['background/'],
        [ttbar]
    )

    bkg = list(_bkg_keys(evaluated, sigkey))
    bkg.remove(ttbar)
    b_scores, b_weights = _get_scores_and_weights(
        evaluated[sigkey + '/background/'],
        data['background/'],
        bkg
    )

    scores = np.concatenate([s_scores, tt_scores, b_scores])
    weights = np.concatenate([s_weights, tt_weights, b_weights])
    labels = np.concatenate([
        np.ones_like(s_scores),
        np.zeros_like(tt_scores),
        np.zeros_like(b_scores)
    ])

    fpr, tpr, thr = sklearn.metrics.roc_curve(
        y_true=labels,
        y_score=scores,
        sample_weight=weights
    )

    zs = significance(
        lumi * tpr * np.sum(s_weights),
        lumi * fpr * (np.sum(b_weights) + np.sum(tt_weights)),
        uncert
    )

    max_z = -float('inf')
    i_best = None
    for i in range(zs.shape[0]):
        if zs[i] > max_z:
            w_sel = tt_weights[np.where(tt_scores >= thr[i])]
            w_sum = np.sum(w_sel)
            if w_sum > 0:
                if np.sqrt(np.sum(w_sel * w_sel)) / w_sum <= 0.3:
                    max_z = zs[i]
                    i_best = i

    if return_effs:
        return thr[i_best], fpr[i_best], tpr[i_best]
    return thr[i_best]



def compute_yield(scores, weights, threshold):
    """ Compute yield for a luminosity of 1 fb^-1
    Arguments:
      -- scores: the model's output, event-by-event
      -- weights: events-wise 1 fb^-1 weight
      -- threshold: the decision threshold
    Returns: The 1 fb^-1 yield
    """
    logging.debug('scores: %s, weights: %s', scores.shape, weights.shape)
    return np.sum(weights[np.where(scores >= threshold)])


def compute_significance_grid(evaluated, data, lumi, uncert, threshold_ttbar_only=False):
    """ Compute significance across the whole mass grid

    Arguments:
      -- evaluated: H5 dataset containing the output probabilities for
         all mass points
      -- data: H5 containing the input data, to get the weights
      -- lumi: target luminosity
      -- uncert: systematic uncertainty used in significance computation
      -- threshold_ttbar_only: only consider the ttbar
         stat. uncert. critarion for determining the decision
         threshold
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
        if threshold_ttbar_only:
            thr = compute_threshold_ttbar(evaluated, sigkey, data)
        else:
            thr = compute_threshold(evaluated, sigkey, data, lumi, uncert)
        s_yield, b_yield, expz = compute_significance(
            evaluated,
            data,
            sigkey,
            thr,
            lumi,
            uncert
        )

        fields = sigkey.split('_')
        results[i] = (
            int(fields[1]),
            int(fields[3]),
            expz,
            s_yield,
            b_yield
        )
        logging.info('%s: %f', sigkey, expz)

    return results


def compute_significance(evaluated, data, sigkey, threshold, lumi, uncert):
    s_yield = compute_yield(
        scores=evaluated[sigkey + '/signal/' + sigkey].value[:, 0],
        weights=data['signal/' + sigkey + '/metadata'].value['M_weight'],
        threshold=threshold
    )

    b_yield = 0
    for bkgkey in _bkg_keys(evaluated, sigkey):
        logging.debug('  %s', bkgkey)
        b_yield += compute_yield(
            scores=evaluated[sigkey + '/background/' + bkgkey].value[:, 0],
            weights=data['background/' + bkgkey + '/metadata'].value[
                'M_weight'
            ],
            threshold=threshold
        )

    b_yield *= lumi
    s_yield *= lumi

    expz = significance(
        s_yield,
        b_yield,
        uncert
    )

    return s_yield, b_yield, expz


def compute_n_excluded(grid):
    """ Compute the number of excluded points at 95% CL """
    return np.count_nonzero(grid[np.where(grid['z'] >= 1.64)])


def _mbj_hist_to_grid(mbj_hist):
        masses = []
        for i_x in range(1, mbj_hist.GetXaxis().GetNbins() + 1):
            for i_y in range(1, mbj_hist.GetYaxis().GetNbins() + 1):
                mg = mbj_hist.GetXaxis().GetBinLowEdge(i_x)
                ml = mbj_hist.GetYaxis().GetBinLowEdge(i_y)
                z = mbj_hist.GetBinContent(i_x, i_y)
                masses.append((mg, ml, z))
        return np.array(masses, dtype=[('mg', 'i4'), ('ml', 'i4'), ('z', 'f4')])


def compute_exclusion_above_mbj(mbj_hist, model_grid):
    mbj_grid = _mbj_hist_to_grid(mbj_hist)
    cnt = 0
    for i in range(mbj_grid.shape[0]):
        mg = mbj_grid['mg'][i]
        ml = mbj_grid['ml'][i]
        z_mbj = mbj_grid['z'][i]
        z_mod = model_grid[np.where((model_grid['mg'] == mg)&(model_grid['ml'] == ml))]['z']
        if z_mod >= 1.64 and z_mbj < 1.64:
            cnt += 1
            logging.debug((mg, ml))
    return cnt


def _bkg_keys(dfile, sigkey):

    for key in dfile[sigkey + '/background']:
        if dfile[sigkey + '/background/' + key].shape[0] > 0:
            yield key


def _get_scores_and_weights(evaluated, data, keys, out_indice=0):
    def _get_one(key):
        scores = evaluated[key].value[:,out_indice]
        weights = data[key + '/metadata'].value['M_weight']
        return scores, weights
    pairs = [_get_one(key) for key in keys]
    scores = np.concatenate([sco for sco, _ in pairs])
    weights = np.concatenate([ws for _, ws in pairs])
    return scores, weights


def _ttbar_key(evaluated, sigkey):
    ttbar = None
    for k in _bkg_keys(evaluated, sigkey):
        if 'ttbar' in k:
            ttbar = k
            break
    return ttbar
