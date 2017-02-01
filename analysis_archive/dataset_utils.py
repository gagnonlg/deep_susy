def get_header_dict(h5file, dsetname):

    if dsetname not in ['inputs', 'labels', 'metadata']:
        raise ValueError("dsetname not in ['inputs', 'labels', 'metadata']")

    idict = {}
    for i, name in enumerate(h5file['header/' + dsetname]):
        idict[name] = i

    return idict


import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import root_numpy


def make_graphs(h5file):

    inputs_dict = get_header_dict(h5file, 'inputs')
    for key in inputs_dict:

        name = 'training/inputs: ' + key
        plt.title(name)
        plt.hist(h5file['training/inputs'][:,inputs_dict[key]], bins=100)
        plt.savefig(key+'.pdf')
        plt.close()

def compare_h5_ttree(h5file, ttree):
    
    inputs_dict = get_header_dict(h5file, 'inputs')

    for key in inputs_dict:

        h5_vect = h5file['training/inputs'][:,inputs_dict[key]]
        tr_vect = root_numpy.tree2array(ttree, branches=key)

        plt.hist(h5_vect, bins=100, normed=True, histtype='step', label='training/inputs')
        plt.hist(tr_vect, bins=100, normed=True, histtype='step', label='NNinput')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.xlabel(key)
        plt.savefig(key+'.pdf')
        plt.close()

