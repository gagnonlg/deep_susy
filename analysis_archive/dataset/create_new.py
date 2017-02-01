import os

import h5py as h5
import root_numpy

def root_to_h5(path):

    array = root_numpy.root2array(path)

    # output path
    outpath = os.path.basename(path)
    if outpath.endswith('.root'):
        outpath = outpath.replace('.root', '.h5')
    else:
        outpath += '.h5'

    h5_file = h5.File(outpath, 'w')
    h5_file.create_dataset(
        'NNinput',
        data=array
    )

    h5_file.close()

    return outpath



def prepare_for_merge(path):

    dsets = ['training', 'validation', 'test']
    outputs = [path.replace('.root', '.{}.root'.format(name) for name in dsets)]
    
    dataset.split(
        root_file,
        'NNinput',
        [0.5, 0.25, 0.25],
        outputs
    )
    
    outputs_h5 = [root_to_h5(path) for path in outputs]

#     __reweight(outputs_h5)
    
# def main():

#     file_list = ...
#     out_dir = ...

#     os.mkdir(out_dir)
#     os.chdir(out_dir)

#     for path in file_list:
#         prepare_for_merge(path)

#     merge(out_dir, out_file)
    
    
# """
# split: coded, tested
# __replace_by_hdf5: coded, tested
# __reweight: TODO
# merge: TODO



# """
