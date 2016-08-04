""" select.py: module to interface with select.cxx event selection code """

import argparse
import logging
import os
import re
import subprocess

import dataset.gtt
import utils

__all__ = ['select']

LOGGER = logging.getLogger('dataset.select')


def select(inputs, output, target, nsmall=10, nlarge=4, nlepton=4,
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
      target: integer value of target class
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

    dsid = __get_dsid(inputs[0])
    if dsid is None:
        m_g = '0'
        m_l = '0'
    else:
        masses = dataset.gtt.get_masses(dsid)
        m_g = str(masses[0])
        m_l = str(masses[1])

    LOGGER.info('parameters set to m_gluino=%s, m_lsp=%s', m_g, m_l)

    subprocess.check_call([
        program,
        output,
        str(nsmall),
        str(nlarge),
        str(nlepton),
        str(met_max),
        str(ht_max),
        str(target),
        m_g,
        m_l
    ] + inputs)


def __get_dsid(path):
    match = re.match(r'.*\.([0-9]+)\.Gtt\.', path)
    if match is not None:
        return int(match.group(1))
    else:
        return None


def get_program_path_():
    """ Returns the select.cxx executable path, regardless of existance """
    return '{}/bin/select'.format(utils.top_directory())


def expand_input_list_(lst):
    """ Expand any comma-separated string of paths into python list """
    new_lst = []
    for path in lst:
        new_lst += path.split(',')
    return new_lst


def main_():
    """ main function if module called as script """
    argp = argparse.ArgumentParser()
    argp.add_argument('--inputs', nargs='+', required=True)
    argp.add_argument('--output', required=True)
    argp.add_argument('--target', required=True, type=int)
    argp.add_argument('--nsmall', type=int, default=10)
    argp.add_argument('--nlarge', type=int, default=4)
    argp.add_argument('--nlepton', type=int, default=4)

    grp = argp.add_mutually_exclusive_group()
    grp.add_argument('--met-filter', default=False, action='store_true')
    grp.add_argument('--ht-filter', default=False, action='store_true')

    args = argp.parse_args()

    select(
        inputs=args.inputs,
        output=args.output,
        target=args.target,
        nsmall=args.nsmall,
        nlarge=args.nlarge,
        nlepton=args.nlepton,
        met_max=(200 if args.met_filter else float('inf')),
        ht_max=(600 if args.ht_filter else float('inf'))
    )

    return 0
