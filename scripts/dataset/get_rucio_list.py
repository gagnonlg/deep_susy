#!/bin/env python2
""" get_rucio_list: program to get list of datasets to fetch with rucio """


import argparse
import re
import subprocess
import sys

import utils

SIGNAL_PATH_ = '{}/signal.list'.format(utils.top_directory())
BKGND_PATH_ = '{}/bkgnd.list'.format(utils.top_directory())


def call_rucio_(search_keys):
    """ get list of dataset from rucio

    Obtain a list of datasets according to a set of search keys using
    rucio list-dids.

    Args:
      search_keys: list of expression passed to rucio list-dids
    Returns:
      list of datasets
    Raises:
      OSError: The rucio executable was not found on PATH, probably because
               'lsetup rucio' was not run.
    """

    dsets = []
    for key in search_keys:
        try:
            output = subprocess.check_output(['rucio', 'list-dids', key])
        except subprocess.CalledProcessError:
            pass

        for line in output.split('\n'):
            fields = line.split('|')
            if len(fields) != 4:
                continue
            if 'SCOPE:NAME' in fields[1]:
                continue
            dsets.append(fields[1].strip())

    return dsets


def dsid_sort_(lst):
    """ sort list of dataset by DSID

    Args:
      lst: list of datasets to sort
    Returns:
      list of tuples (DSID,dataset) sorted by DSID
    """
    to_sort = []
    for line in lst:
        dsid = int(re.match(r'.*\.([0-9]+)\..*\.DAOD_SUSY10.*', line).group(1))
        to_sort.append((dsid, line))
    return sorted(to_sort)


def main_():
    """ main function when called from cli """
    argp = argparse.ArgumentParser()
    argp.add_argument('search_keys', nargs='+')
    args = argp.parse_args()

    signals = open(SIGNAL_PATH_).readlines()
    bkgnds = open(BKGND_PATH_).readlines()

    dset_list = [d.strip('\n') for d in signals + bkgnds]

    lst = [
        n for n in call_rucio_(args.search_keys)
        if any([m in n for m in dset_list])
    ]

    sorted_lst = dsid_sort_(lst)
    names = [t[1] for t in sorted_lst]
    dsids = [t[0] for t in sorted_lst]

    last = dsids[0]
    for dsi in dsids[1:]:
        if dsi == last:
            sys.stderr.write('WARNING: duplicated dsid: %d\n' % dsi)
        last = dsi

    print '\n'.join(names)

    return 0
