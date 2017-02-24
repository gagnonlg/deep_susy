import argparse
import logging
import sqlite3

import numpy as np
import sklearn.mixture

import utils

log = logging.getLogger('significance_threshold')

def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('--db', required=True)
    argp.add_argument('--mixtures', type=int, default=1)
    return argp.parse_args()

def main():
    args = get_args()

    log.info('Opening %s', args.db)
    db = sqlite3.connect(args.db)

    log.info('Fitting significance with %d gaussians', args.mixtures)
    mixture = sklearn.mixture.GMM(args.mixtures)
    data = np.array(list(db.execute('SELECT significance FROM perf')))
    mixture.fit(data)
    ibest = np.argmax(mixture.means_)
    mu = mixture.means_[ibest]
    sig = mixture.covars_[ibest]
    lower = mu + 1.64 * sig

    log.info('Result (higher gaussian): mu=%.3f, sigma=%.3f', mu, sig)
    log.info('95%% CL lower bound on significance: %.3f', lower)

    tot, = db.execute('SELECT count(*) FROM perf').fetchone()
    pas, = db.execute('SELECT count(*) FROM perf WHERE significance > %f' % lower).fetchone()
    ids = [
        str(id) for id, in
        db.execute('SELECT id FROM perf WHERE significance > %f' % lower).fetchall()
    ]
    log.info('%d significantly best neural networks out of %d', pas, tot)
    log.info('IDs: %s', ','.join(ids))

if __name__ == '__main__':
    utils.main(main, 'significance_threshold')
