""" program to add a model to the hyperparameter database """
import argparse
import logging
import os
import sqlite3

from deep_susy import model, utils

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
    # pylint: disable=too-many-arguments, invalid-name
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


def _insert_model(dirpath, dbpath):

    model_name = os.path.basename(dirpath.rstrip('/'))
    dbs = _get_database(dbpath)

    _check_for_model(model_name, dbs)

    model_path = '{}/{}_trained.h5'.format(dirpath, model_name)
    reach_path = '{}/{}_reach.txt'.format(dirpath, model_name)

    k_model = model.load_keras(model_path, compile=False)

    logging.info('Getting the hyperparameters')
    row = model.get_hyperparameters(k_model)

    logging.info('Getting exclusion data')
    with open(reach_path, 'r') as rfile:
        lines = rfile.readlines()
        n_excl_training = int(lines[0].split(' ')[-1])
        n_excl_valid = int(lines[1].split(' ')[-1])

    logging.info('Updating the database')
    row.update(
        name=model_name,
        n_excluded_training=n_excl_training,
        n_excluded_validation=n_excl_valid
    )

    dbs.execute(_insert_statement(**row))
    dbs.commit()


def _insert_all(directory, dbs):
    for path in os.listdir(directory):
        _insert_model('{}/{}'.format(directory, path), dbs)


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--db', required=True)
    grp = args.add_mutually_exclusive_group(required=True)
    grp.add_argument('--directory')
    grp.add_argument('--model')
    return args.parse_args()


def _main():
    args = _get_args()
    if args.model is not None:
        _insert_model(args.model, args.db)
    else:
        _insert_all(args.directory, args.db)


if __name__ == '__main__':
    utils.main(_main, '')
