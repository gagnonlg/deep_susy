import argparse
import h5py as h5
import numpy as np
import model
import os
import tempfile


args = argparse.ArgumentParser()
args.add_argument('defn')
args.add_argument('model')
args.add_argument('norm')
args.add_argument('data')
args = args.parse_args()

if not os.path.exists(args.norm):
    norm = np.array([0,1])
    tmp = tempfile.NamedTemporaryFile()
    np.savetxt(tmp.name, norm)
    norm = tmp.name
else:
    norm = args.norm
    tmp = None

data = h5.File(args.data, 'r')
dataX = np.array(data['validation/inputs'])
dataY = np.array(data['validation/labels'])
netw = model.TrainedModel.from_files(args.defn, args.model, norm)
metrics = netw.evaluate(dataX, dataY).save()

if tmp is not None:
    tmp.close()
