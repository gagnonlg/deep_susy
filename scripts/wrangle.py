""" Wrangle the data from an optimization round into a database """

import argparse
from glob import glob
import os
import sqlite3

import h5py as h5

import model
import utils


def __create_database(path):
    database = sqlite3.connect(path)
    with open(utils.project_path('sql/CREATE_TABLE_perf.sql')) as sqlfile:
        sql = sqlfile.read().strip()
    database.execute(sql)
    database.commit()
    return database


def __insert_model(database, definition_path, metrics_path, jobid):

    definition = model.ModelDefinition.from_file(definition_path)

    with h5.File(metrics_path) as metf:
        auc = metf['auc'].value

    with open(utils.project_path('sql/INSERT_INTO_perf.sql.template')) as tfl:
        template = tfl.read().strip()

    sql = template.format(
        jobid=jobid,
        l2_reg=definition.l2_reg,
        learning_rate=definition.learning_rate,
        max_epochs=definition.max_epochs,
        min_epochs=definition.min_epochs,
        momentum=definition.momentum,
        n_hidden_layers=definition.n_hidden_layers,
        n_hidden_units=definition.n_hidden_units,
        normalize=int(definition.normalize),
        patience=definition.patience,
        reweight=int(definition.reweight),
        auc=auc
    )

    database.execute(sql)
    database.commit()


def __main():
    args = argparse.ArgumentParser()
    args.add_argument('--inputs', required=True, nargs='+')
    args.add_argument('--output', required=True)
    args = args.parse_args()

    names = [os.path.basename(dirpath) for dirpath in args.inputs]
    defs = [glob(dirpath + '/*.definition.txt')[0] for dirpath in args.inputs]
    metrics = [glob(dirpath + '/*.metrics.h5')[0] for dirpath in args.inputs]

    database = __create_database(args.output)

    for name, defp, metp in zip(names, defs, metrics):
        print '==> Inserting ' + name
        __insert_model(database, defp, metp, name.split('.')[0])


if __name__ == '__main__':
    __main()
