""" Functions for k-fold cross validation """
import logging

import numpy as np

LOGGER_ID = 0


def k_fold(data, fit_function, nfolds=2):
    """ K-fold cross validation of arbitrary model fitting strategy

    Arguments:
        data -- tuple (x_data, y_data) on which to fit the models
        fit_function -- function accepting data tuple and returning a
                        fitted model
        nfolds -- number of folds (`k` in k-folds)

    Returns:
        A tuple (fitted model, metrics)
    """

    # Increase LOGGER_ID by one such that the message source can be identified
    # when k_fold() is nested inside fit_function
    global LOGGER_ID  # pylint: disable=global-statement
    LOGGER_ID += 1
    logger = logging.getLogger('k-fold #{}'.format(LOGGER_ID))

    # K training done here
    metrics = []
    for i, (train_fold, val_fold) in enumerate(__partition(data, nfolds)):
        logger.info('Fold %d/%d', i, nfolds)
        logger.info('Running fit_function')
        model = fit_function(train_fold)
        logger.info('Testing model')
        metrics.append(model.test(val_fold))

    # k-fold cross-validation yields an estimation of the performance
    # of the fit strategy when evaluated on unseen data
    logger.info('Averaging metrics across all folds')
    final_metrics = __average_metrics(metrics)
    logger.info('mean: %s', final_metrics[0])
    logger.info('std: %s', final_metrics[1])

    # To actually achieve this performance, the model must be trained
    # on all available data
    logger.info('Training final model')
    final_model = fit_function(data)

    return final_model, final_metrics


def __partition(data, nfolds):

    # Pre-condition checks

    if nfolds <= 0:
        raise ValueError("number of folds must be > 0")

    x_data, y_data = data

    if not isinstance(x_data, np.ndarray) or \
       not isinstance(y_data, np.ndarray):
        raise TypeError('data must have type (np.ndarray, np.ndarray)')

    n_data = x_data.shape[0]

    if n_data != y_data.shape[0]:
        raise ValueError('X and Y data Shape mismatch')

    if n_data < nfolds:
        raise ValueError('not enough data for requested number of folds')

    # define the slices
    test_fold_size = n_data / nfolds
    start_indices = np.arange(0, n_data, test_fold_size)
    stop_indices = start_indices + test_fold_size

    # When the dataset size is not a multiple of the number of folds,
    # ensure we return the requested number of folds with the
    # potential elements in excess in the last fold
    if start_indices.size > nfolds:
        start_indices = start_indices[:nfolds]
        stop_indices = stop_indices[:nfolds]
        stop_indices[-1] = n_data

    for i_test_start, i_test_stop in zip(start_indices, stop_indices):

        # The training fold is in general distributed before and after
        # the test fold
        x_train_fold = np.append(x_data[0:i_test_start], x_data[i_test_stop:])
        y_train_fold = np.append(y_data[0:i_test_start], y_data[i_test_stop:])

        # The test fold is always contiguous
        x_test_fold = x_data[i_test_start:i_test_stop]
        y_test_fold = y_data[i_test_start:i_test_stop]

        yield (x_train_fold, y_train_fold), (x_test_fold, y_test_fold)


def __average_metrics(metrics):
    metrics_2d = np.array(metrics)
    return np.mean(metrics_2d, axis=0), np.std(metrics_2d, axis=0, ddof=1)
