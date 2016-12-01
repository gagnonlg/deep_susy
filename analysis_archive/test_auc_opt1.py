import glob
import re

import h5py
import keras
import numpy as np

import roc

input_file = h5py.File('gtt_deep_learning_dataset_0.h5', 'r')
x_valid = np.array(input_file['validation']['inputs'])
y_valid = np.array(input_file['validation']['labels'])

m_valid = input_file['validation']['metadata']
m_header = input_file['header']['metadata']

# TODO find better way
weights = np.array(m_valid[:,0])

i_sorted_labels = np.argsort(y_valid)
x_valid = x_valid[i_sorted_labels]
y_valid = y_valid[i_sorted_labels]
weights = weights[i_sorted_labels]

normalization = np.loadtxt('norm.txt')

x_valid -= normalization[0]
x_valid /= normalization[1]

with open('AUC.txt', 'w') as auc_file:
    for path in glob.glob('fit_model_*.hdf5'):
        id = re.match(r'fit_model_(.*)\.hdf5', path).group(1)
        print "=> {}".format(id)
        model = keras.models.load_model(path)
        scores = model.predict(x_valid)[:,0]
        fp, tp = roc.roc_curve(scores, y_valid, weights)
        auc = roc.auc(fp, tp)
        print auc
        auc_file.write('{} {}\n'.format(auc, id))

    
with open('AUC.txt', 'r') as auc_file:
    aucs = []
    for line in auc_file:
        fields = line.strip().split()
        aucs.append(float(fields[0]))

aucs = np.array(aucs)
mu = np.mean(aucs)
sig = np.std(aucs, ddof=1)

print mu
print sig
print sig*1.64

print np.count_nonzero(np.where(aucs > mu+(sig*1.64)))

def normal(x,mu,sig):
    norm = 1.0/np.sqrt(2*sig*sig*np.pi)
    return norm * np.exp(-((x - mu)**2)/(2*sig*sig))


import matplotlib.pyplot as plt

plt.hist(aucs, normed=True, label='Neural networks')
x = np.linspace(np.min(aucs), np.max(aucs), 1000)
plt.plot(x, normal(x, mu, sig), label='Normal(mu={:.4f}, sig={:.4f})'.format(mu, sig))
plt.plot([mu+sig*1.64, mu+sig*1.64], [0, normal(mu+sig*1.64, mu, sig)], '--', label='95% upper bound')
plt.xlabel('AUC')
plt.ylabel('NN density')
plt.legend(loc='best')
plt.axis([0.9976, 0.9990, 0, 3000])
plt.savefig('aucs.png')
