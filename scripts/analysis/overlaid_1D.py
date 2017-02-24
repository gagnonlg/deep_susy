import argparse
import logging
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

import utils

log = logging.getLogger('overlaid_1D')

def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('--db', required=True)
    return argp.parse_args()

def make_query(cond):
    sql = 'SELECT significance FROM perf'
    return sql + ('' if cond is None else ' WHERE ' + cond)

def query_significance(db, conditions):
    return [np.array(list(db.execute(sql))) for sql in
            [make_query(cnd) for cnd in conditions]]

def save_hist(db, conditions, path):
    for distr, lbl in zip(query_significance(db, conditions), conditions):
        plt.hist(distr, histtype='step', bins=15, label=lbl)
    plt.xlabel('significance at min. bkg efficiency')
    plt.ylabel('Number of neural networks')
    if any(conditions):
        plt.legend(loc='best')
    plt.savefig(path)
    plt.close()
    log.info('Created %s', path)


def main():
    args = get_args()
    db = sqlite3.connect(args.db)

    save_hist(
        db,
        conditions=[None],
        path='significance.png'
    )

    save_hist(
        db,
        conditions=['reweight=0', 'reweight=1'],
        path='significance_reweight.png',
    )

    save_hist(
        db,
        conditions=['normalize=0', 'normalize=1'],
        path='significance_normalize.png',
    )
    
    save_hist(
        db,
        conditions=[
            "early_stop_metric='loss'",
            "early_stop_metric='precision'",
            "early_stop_metric='recall'",
            "early_stop_metric='fmeasure'",
        ],
        path='significance_early_stop_metric.png',
    )



    

if __name__ == '__main__':
    utils.main(main, 'overlaid_1D')
