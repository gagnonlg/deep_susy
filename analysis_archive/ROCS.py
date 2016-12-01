import time

import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np

def roc_curve(scores, weights, labels):

    i_sorted = np.argsort(scores)[::-1]

    w_false_positive = 0
    w_true_positive = 0
    false_positives = []
    true_positives = []

    f_prev = - float('inf')

    for c,i in enumerate(i_sorted):
        f_score = scores[i]
        
        if f_score != f_prev:
            false_positives.append(w_false_positive)
            true_positives.append(w_true_positive)
            f_prev = f_score

        if labels[i] == 1:
            w_true_positive += weights[i]
        else:
            w_false_positive += weights[i]


    false_positives.append(w_false_positive)
    true_positives.append(w_true_positive)
            
            
    true_positives = np.array(true_positives)
    true_positives /= np.sum(weights[np.where(labels == 1)])

    false_positives = np.array(false_positives)
    false_positives /= np.sum(weights[np.where(labels == 0)])

    return false_positives, true_positives

def roc_curve_vectorized(scores, weights, labels):

    # First, sort everything in order of decreasing score
    i_sorted = np.argsort(scores)[::-1]
    scores_sorted = scores[i_sorted]
    weights_sorted = weights[i_sorted]
    labels_sorted = labels[i_sorted]

    # Compute the cumulative sum of positive and negative examples
    wsum_pos = np.cumsum(np.where(labels_sorted == 1, weights_sorted, 0))
    wsum_neg = np.cumsum(np.where(labels_sorted == 0, weights_sorted, 0))

    """ Correctly handle successive example with equal score. 
    
    We want a
    decreasing score so the resulting array could be flipped. The
    problem is that np.unique keeps only the first value of a run of
    equal examples, so when the array is flipped, the behavior is
    reversed.

    as an example, imagine we have the following array, sorted in
    decreasing order: 
    [d,c1,c2,b,a] where c1 == c2 
    
    indice sequence we want: [0,1,3,4]
    
    np.unique will sort the array:
    [d,c1,c2,b,a] -> [a,b,c1,c2,d]
    indices returned: [0,1,2,4]

    If we use the negative of the array, it becomes
    [-d,-c1,-c2,-b,-a]
    which is already sorted. the indices returned are:
    [0,1,3,4]
    as wanted :)

    """

    _, i_uniq = np.unique(np.negative(scores_sorted), return_index=True)

    # then we can compute the actual rates
    true_positives = wsum_pos[i_uniq] / wsum_pos[-1]
    false_positives = wsum_neg[i_uniq] / wsum_neg[-1]

    # append zero to beginning to have correct ROC curve
    return np.append(0, false_positives), np.append(0, true_positives)

def main():

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

    model = keras.models.load_model('fit_model.hdf5')
    normalization = np.loadtxt('norm.txt')

    scores = model.predict((x_valid - normalization[0]) / normalization[1])
    scores = scores[:,0]

    print "=> roc_curve"
    t0 = time.time()
    fp1, tp1 = roc_curve(scores, weights, y_valid)
    print time.time() - t0
    print "=> roc_curve_vectorized"
    t0 = time.time()
    fp2, tp2 = roc_curve_vectorized(scores, weights, y_valid)
    print time.time() - t0


    plot = plt.semilogx
    #plt.axis([1e-4,1,0,1.1])

    plot(fp1, tp1, label='NN, AUC={}'.format(np.trapz(tp1,fp1)))
    plot(fp2, tp2, label='NNv, AUC={}'.format(np.trapz(tp2,fp2)))

    # random
    line = np.linspace(0,1,1000)
    plot(line,line, label='random, AUC=0.5')

    plt.legend(loc='best')
    plt.xlabel('Fake rate')
    plt.ylabel('Signal Efficiency')

    plt.savefig('test.roc.png')

    
main()
