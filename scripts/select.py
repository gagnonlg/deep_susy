""" select.py: module to interface with select.cxx event selection code """

import argparse
import os
import subprocess

__all__ = ['select']


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

    subprocess.check_call([
        program,
        output,
        str(nsmall),
        str(nlarge),
        str(nlepton),
        str(met_max),
        str(ht_max)
    ] + inputs)


def get_program_path_():
    """ Returns the select.cxx executable path, regardless of existance """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fields = script_dir.split('/')
    fields[-1] = 'bin/select'
    return '/'.join(fields)


def main_():
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
        output=args.output,
        nsmall=args.nsmall,
        nlarge=args.nlarge,
        nlepton=args.nlepton,
        met_max=(200 if args.met_filter else float('inf')),
        ht_max=(600 if args.ht_filter else float('inf'))
    )

if __name__ == '__main__':
    main_()
