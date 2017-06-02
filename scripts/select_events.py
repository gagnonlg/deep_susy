#!/bin/env python2
""" executable cli script for select.py """

import argparse
import logging
import os
import re
import subprocess

import gtt
import utils

__all__ = ['select']

LOGGER = logging.getLogger('dataset.select')


def select(inputs, output, nsmall=10, nlarge=4, nlepton=4,
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


def output_path(output):
    """ Add version with git describe """

    if output.endswith('.root'):
        output = output.replace('.root', '')

    repo = utils.top_directory()

    cmd = "cd {} && git describe $(git log -n1 --pretty=%h {})"
    v1 = subprocess.check_output(
        cmd.format(
            repo,
            repo + '/scripts/select_events.py'
        ),
        shell=True
    )
    v2 = subprocess.check_output(
        cmd.format(
            repo,
            repo + '/src/select.cxx'
        ),
        shell=True
    )

    nv1 = [float(f) for f in v1.split('-')[:2]]
    nv2 = [float(f) for f in v2.split('-')[:2]]

    if nv1 < nv2:
        ver = v2[:-1]
    else:
        ver = v1[:-1]

    mods = subprocess.check_output(
        'git diff-index --name-only HEAD',
        shell=True
    )

    if len(mods) > 0:
        ver += '-M'

    return '{}.NNinput.{}.root'.format(output, ver)


def select_main():
    """ main function if module called as script """
    argp = argparse.ArgumentParser()
    argp.add_argument('--inputs', nargs='+', required=True)
    argp.add_argument('--output', required=True)
    argp.add_argument('--nsmall', type=int, default=10)
    argp.add_argument('--nlarge', type=int, default=4)
    argp.add_argument('--nlepton', type=int, default=4)

    grp = argp.add_mutually_exclusive_group()
    grp.add_argument('--met-filter', default=False, action='store_true')
    grp.add_argument('--ht-filter', default=False, action='store_true')

    args = argp.parse_args()

    select(
        inputs=args.inputs,
        output=output_path(args.output),
        nsmall=args.nsmall,
        nlarge=args.nlarge,
        nlepton=args.nlepton,
        met_max=(200 if args.met_filter else float('inf')),
        ht_max=(600 if args.ht_filter else float('inf'))
    )

    return 0


if __name__ == '__main__':
    utils.main(select_main, 'select')
