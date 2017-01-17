""" Functions to downsample a root file """
import argparse
import root_numpy


def downsample(infile, outfile, nkeep):
    """ infile: the root file to downsample
        outfile: where the result will be stored
        nkeep: number of events to keep
    """
    root_numpy.array2root(
        arr=root_numpy.root2array(
            filenames=infile,
            treename='NNinput',
            stop=nkeep
        ),
        filename=outfile,
        treename='NNinput',
        mode='RECREATE'
    )


def downsample_main():
    """ cli interface """
    args = argparse.ArgumentParser()
    args.add_argument('--infile', required=True)
    args.add_argument('--outfile', required=True)
    args.add_argument('--nkeep', type=int, required=True)
    args = args.parse_args()

    downsample(args.infile, args.outfile, args.nkeep)

    return 0
