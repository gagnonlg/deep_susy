#!/bin/env python2
""" executable cli script for select.py """

import argparse
import logging
import os
import subprocess

from deep_susy import utils

__all__ = ['select']

LOGGER = logging.getLogger('dataset.select')


def select(inputs, output, dsid, nsmall=10, nlarge=4, nlepton=4,
           met_max=float('inf'), ht_max=float('inf')):  \
           # pylint: disable=too-many-arguments
    """Runs the event selection code on specified inputs.

    Runs the 'select.cxx' code on inputs with the given
    configuration. The code performs a basic event selection, computes
    event weights, and outputs a flat rootfile with the requested
    number of objects and some metadata. See the 'select.cxx'[0] source
    for details.

    Args:
      inputs: list of paths to MBJ ntuples
      output: path to output root file. Must not already exist.
      dsid: the dataset id of the input
      nsmall: (optional) number of small-R jets in output
      nlarge: (optional) number of large-R jets in output
      nlepton: (optional) number of leptons in output
      met_max: (optional) met filter cut
      ht_max: (optional) ht filter cut

    Return value:
        None

    Raises:
        RuntimeError: Executable not found
        CalledProcessError: select did not succeed

    References
    ----------
    [0] https://gitlab.cern.ch/lgagnon/deep_susy/blob/master/src/select.cxx

    """
    program = get_program_path_()
    if not os.path.exists(program):
        raise RuntimeError('{} not found'.format(program))

    inputs = expand_input_list_(inputs)

    subprocess.check_call([
        program,
        output,
        str(nsmall),
        str(nlarge),
        str(nlepton),
        str(met_max),
        str(ht_max),
        dsid,
    ] + inputs)


def get_program_path_():
    """ Returns the select.cxx executable path, regardless of existance """
    return '{}/bin/select'.format(utils.top_directory())


def expand_input_list_(lst):
    """ Expand any comma-separated string of paths into python list """
    new_lst = []
    for path in lst:
        new_lst += path.split(',')
    return new_lst


def output_path(output, data_version, suffix):
    """ Add version with git describe """

    if output.endswith('.root'):
        output = output.replace('.root', '')

    repo = utils.top_directory()

    ver = subprocess.check_output(
        "cd {} && git describe".format(repo),
        shell=True
    ).strip()

    return '{}.NNinput.{}.{}.{}.root'.format(
        output,
        data_version,
        ver,
        suffix
    )


def get_filters(dsid):
    """ Get met or ht cut accoring to DSID """
    htf = metf = float('inf')
    if dsid in ['410000', '410004']:
        htf = 600.0
    elif dsid in ['410013', '410014']:
        metf = 200.0
    return metf, htf


def select_main():
    """ main function if module called as script """
    argp = argparse.ArgumentParser()
    argp.add_argument('--inputs', nargs='+', required=True)
    argp.add_argument('--dsid', required=True)
    argp.add_argument('--nsmall', type=int, default=10)
    argp.add_argument('--nlarge', type=int, default=4)
    argp.add_argument('--nlepton', type=int, default=4)
    argp.add_argument('--data-version')
    args = argp.parse_args()

    metf, htf = get_filters(args.dsid)

    suffix = '{}-{}-{}'.format(args.nsmall, args.nlarge, args.nlepton)

    select(
        inputs=args.inputs,
        output=output_path(args.dsid, args.data_version, suffix),
        nsmall=args.nsmall,
        nlarge=args.nlarge,
        nlepton=args.nlepton,
        met_max=metf,
        ht_max=htf,
        dsid=args.dsid
    )

    return 0


if __name__ == '__main__':
    utils.main(select_main, 'select')
