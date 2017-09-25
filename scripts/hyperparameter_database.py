""" program to add a model to the hyperparameter database """
import argparse
import glob
import logging
import os
import sqlite3

import h5py as h5

from deep_susy import evaluation, model, utils

CREATE_TEMPLATE = open(
    utils.project_path('sql/CREATE_TABLE_perf.sql'),
    'r'
).read().strip()


INSERT_TEMPLATE = open(
    utils.project_path('sql/INSERT_INTO_perf.sql.template'),
    'r'
).read().strip()


def _insert_statement(name,
                      n_hidden_layers,
                      n_hidden_units,
                      normalization,
                      l2,
                      n_excluded_training,
                      n_excluded_validation):
    return INSERT_TEMPLATE.format(
        name='"' + name + '"',
        n_hidden_layers=n_hidden_layers,
        n_hidden_units=n_hidden_units,
        normalization=normalization,
        l2=l2,
        n_excluded_training=n_excluded_training,
        n_excluded_validation=n_excluded_validation
    )


def _get_database(path):
    dbs = sqlite3.connect(path)
    dbs.execute(CREATE_TEMPLATE)
    return dbs


def _check_for_model(name, dbs):
    res = dbs.execute("SELECT * FROM perf WHERE name='%s'" % name)
    if res.fetchone() is not None:
        raise RuntimeError('Model %s already in database' % name)


def _insert_model(dirpath, datapath, dbpath):

    model_name = os.path.basename(dirpath.rstrip('/'))
    dbs = _get_database(dbpath)

    _check_for_model(model_name, dbs)

    model_path = glob.glob('{}/*_trained.h5'.format(dirpath))[0]
    training_path = glob.glob('{}/*_evaluated-training.h5'.format(dirpath))[0]
    validation_path = glob.glob(
        '{}/*_evaluated-validation.h5'.format(dirpath)
    )[0]

    k_model = model.load_keras(model_path)
    training_set = h5.File(training_path, 'r')
    validation_set = h5.File(validation_path, 'r')
    data = h5.File(datapath, 'r')

    logging.info('Getting the hyperparameters')
    row = model.get_hyperparameters(k_model)
    logging.info('Computing the exclusion for the training set')
    n_excl_training = evaluation.compute_n_excluded(
        evaluation.compute_significance_grid(
            evaluated=training_set,
            data=data['training'],
            lumi=36.1,
            uncert=0.3
        )
    )
    logging.info('Computing the exclusion for the validation set')
    n_excl_valid = evaluation.compute_n_excluded(
        evaluation.compute_significance_grid(
            evaluated=validation_set,
            data=data['validation'],
            lumi=36.1,
            uncert=0.3
        )
    )
    row.update(
        name=model_name,
        n_excluded_training=n_excl_training,
        n_excluded_validation=n_excl_valid
    )

    dbs.execute(_insert_statement(**row))
    dbs.commit()


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', required=True)
    args.add_argument('--data', required=True)
    args.add_argument('--db', required=True)
    return args.parse_args()


def _main():
    args = _get_args()
    _insert_model(args.model, args.data, args.db)


if __name__ == '__main__':
    utils.main(_main, '')
