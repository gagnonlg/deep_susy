import numpy as np


def roc_curve(scores, labels, weights=None):
    """ Vectorized implementation of ROC algorithm

    Arguments:
      scores: array of  scores
      labels: array of binary labels (1 == positive, 0 == negative)
      weights: (optional) array of weights for each example (defaults to 1)
    Returns:
      tuple of false positive rates, true positive rates
    Throws:
      ValueError if the array's shape are mismatched

    Ref: Fawcett T. An introduction to ROC analysis. Pattern
        Recognition Letters. 2006 Jun;27(8):p.866
    """

    # Assign default weight of 1
    if weights is None:
        weights = np.ones_like(scores)

    # ensure the shapes match
    if not scores.shape == labels.shape == weights.shape:
        raise ValueError("mismatched array shapes")

    # Begin to compute the ROC curve.
    # First, sort everything in order of decreasing score
    i_sorted = np.argsort(scores)[::-1]
    scores_sorted = scores[i_sorted]
    weights_sorted = weights[i_sorted]
    labels_sorted = labels[i_sorted]

    # Compute the cumulative sum of positive and negative examples
    wsum_pos = np.cumsum(np.where(labels_sorted == 1, weights_sorted, 0))
    wsum_neg = np.cumsum(np.where(labels_sorted == 0, weights_sorted, 0))

    # Correctly handle successive example with equal score.
    #
    # numpy.unique does what we what, however it returns the indices
    # in a sorted array. at this point we already have an array sorted
    # in decreasing order so the resulting array could be flipped. The
    # problem is that np.unique keeps only the first value of a run of
    # equal examples, so when the array is flipped, the behavior is
    # reversed.
    #
    # as an example, imagine we have the following array, sorted in
    # decreasing order:
    # [d,c1,c2,b,a] where c1 == c2
    #
    # indice sequence we want: [0,1,3,4]
    #
    # np.unique will sort the array:
    # [d,c1,c2,b,a] -> [a,b,c1,c2,d]
    # indices returned: [0,1,2,4]
    #
    # If we use the negative of the array, it becomes
    # [-d,-c1,-c2,-b,-a]
    # which is already sorted. the indices returned are:
    # [0,1,3,4]
    # as wanted.
    #
    # See reference in function's docstring for more information on
    # why we handle successive equally-scored examples this way.
    _, i_uniq = np.unique(np.negative(scores_sorted), return_index=True)

    # then we can compute the actual rates
    true_positives = wsum_pos[i_uniq] / wsum_pos[-1]
    false_positives = wsum_neg[i_uniq] / wsum_neg[-1]

    # append zero to beginning to have correct ROC curve
    return np.append(0, false_positives), np.append(0, true_positives)


def auc(roc):
    fp, tp = roc
    return np.trapz(tp, fp)
