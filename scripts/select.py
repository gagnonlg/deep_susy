import argparse
import os
import subprocess

__all__ = ['select']

def get_program_path_():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fields = script_dir.split('/')
    fields[-1] = 'bin/select'
    return '/'.join(fields)

def select(inputs, output, nsmall=10, nlarge=4, nlepton=4,
           met_max=float('inf'), ht_max=float('inf')):

    program = get_program_path_()
    if not os.path.exists(program):
        raise RuntimeException('{} not found'.format(program))

    subprocess.call([
        program,
        output,
        str(nsmall),
        str(nlarge),
        str(nlepton),
        str(met_max),
        str(ht_max)
    ] + inputs
    )

def main_():

    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--nsmall', type=int, default=10)
    p.add_argument('--nlarge', type=int, default=4)
    p.add_argument('--nlepton', type=int, default=4)

    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--met-filter', default=False, action='store_true')
    grp.add_argument('--ht-filter', default=False, action='store_true')

    args = p.parse_args()

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
