
import h5py as h5
import keras
import matplotlib.pyplot as plt
import numpy as np
from ROOT import RooStats
import scipy.stats
import sklearn.metrics

model_path = "/lcg/storage15/atlas/gagnon/work/5781.atlas11.lps.umontreal.ca_optimization/GttNN_1.keras_model.h5"

norm_path  = "/lcg/storage15/atlas/gagnon/work/5781.atlas11.lps.umontreal.ca_optimization/GttNN_1.normalization.txt"

data_path = "/lcg/storage15/atlas/gagnon/data/NNinput-a97618a.h5"

###

dset = h5.File(data_path)

testX = np.array(dset['validation/inputs'])
testY = np.array(dset['validation/labels'])
testM = np.array(dset['validation/metadata'])

model = keras.models.load_model(model_path)
norm = np.loadtxt(norm_path)
mean = norm[0]
std = norm[1]

scores = model.predict((testX - mean) / std)
#scores = model.predict(testX)

auc = sklearn.metrics.roc_auc_score(testY, scores)
acc = sklearn.metrics.accuracy_score(testY, scores > 0.5)

#print 'auc: {}, acc: {}'.format(auc, acc)

scores_pos = scores[np.where(testY == 1)]
scores_neg = scores[np.where(testY == 0)]

plt.hist(scores_pos, histtype='step', bins=15, normed=True, label='signal')
plt.hist(scores_neg, histtype='step', bins=15, normed=True, label='background')
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel('NN output')
plt.ylabel('Event density')
plt.savefig('NN_output.png')
plt.close()

############

m_header = dset['header/metadata']
mdict = {}
for i, key in enumerate(m_header):
    mdict[key] = i

# sort_key = np.argsort(testY)
# weights_sorted = testM[:,mdict['M_weight']][sort_key]
# scores_sorted = scores[sort_key]
# testY_sorted = testY[sort_key]

# thresholds = []
# significances = []
# for thr in np.linspace(0, 1, 100):
#     print '** {:.2f}'.format(thr)
#     scores_ = scores_sorted >= thr
#     ids = np.where((testY_sorted == 1)&(scores_))[0]
#     idb = np.where((testY_sorted == 0)&(scores_))[0]
#     s = np.sum(weights_sorted[ids])
#     b = np.sum(weights_sorted[idb])
#     thresholds.append(thr)
#     significances.append(ROOT.RooStats.NumberCountingUtils.BinomialExpZ(s, b, 0.3))

# plt.plot(thresholds, significances)
# plt.savefig('significance.png')
# plt.close()
    


############

i_header = dset['header/inputs']
idict = {}
for i, key in enumerate(i_header):
    idict[key] = i

m_header = dset['header/metadata']
mdict = {}
for i, key in enumerate(m_header):
    mdict[key] = i

SR_Gtt_0l_A = (
    (testM[:, mdict['M_nlepton']] == 0) &
    (testM[:, mdict['M_mtb']] > 80) &
    (testM[:, mdict['M_nb']] >= 3) &
    (testX[:, idict['I_met_mag']] > 350) &
    (testM[:, mdict['M_meff']] > 2600) &
    (testM[:, mdict['M_mjsum']] > 300)
)

weights = testM[:, mdict['M_weight']]

s_sel = np.where(SR_Gtt_0l_A & (testY == 1))[0]
b_sel = np.where(SR_Gtt_0l_A & (testY == 0))[0]

w_s_sel = np.sum(weights[s_sel])
w_b_sel = np.sum(weights[b_sel])

stot = np.sum(testM[:, mdict['M_weight']][np.where(testY == 1)])

btotsel = weights[np.where(testY == 0)]
btot = np.sum(btotsel)
btot2 = np.sum(btotsel * btotsel)

print w_s_sel
print w_b_sel

nsig0 = np.count_nonzero(testY)

Aseff = w_s_sel / stot
Abeff = w_b_sel / btot

fpr, tpr, thr = sklearn.metrics.roc_curve(testY, scores, sample_weight=weights, drop_intermediate=False)
plt.plot(fpr, tpr, label='NN')
xrand = np.linspace(1e-5,1,100)
plt.plot(xrand, xrand, '--', label='random')
plt.xscale('log')
plt.axis([1e-5, 1, 0, 1])
plt.ylabel('Signal efficiency')
plt.xlabel('Background efficiency')
plt.text(2e-5, 0.9, 'AUC: {:.3f}'.format(auc))
plt.plot([Abeff], [Aseff], 'o', label='SR_Gtt_0l_A')
plt.legend(loc='best')
plt.savefig('roc.png')
plt.close()


print '** sign'
l = 35
zvalues = [RooStats.NumberCountingUtils.BinomialExpZ(l*s*stot, l*b*btot, 0.3) for s,b in zip(tpr, fpr)]
plt.plot(thr, zvalues)

zsr = RooStats.NumberCountingUtils.BinomialExpZ(l*w_s_sel, l*w_b_sel, 0.3)
plt.plot([0,1], [zsr, zsr])
plt.axis([0.8, 1, 0, 10])


fpr2, tpr2, thr2 = sklearn.metrics.roc_curve(testY, scores, sample_weight=(weights*weights),drop_intermediate=False)

buncert = np.sqrt(fpr2 * btot2) / (fpr * btot)

ifirstb = np.where(buncert > 0.3)[0][-1]
plt.plot([thr[ifirstb], thr[ifirstb]],[0, 10], 'r--')

ifirsts = np.where((l * tpr *  stot) >= 2)[0][0]
print thr[ifirsts]
plt.plot([thr[ifirsts], thr[ifirsts]],[0, 10], 'b--')



plt.savefig('sign.png')

plt.close()



exit()


#############################################


i_header = dset['header/inputs']

rvalues = []
pvalues = []
names = []

for i, name in enumerate(i_header):
    y = testX[:,i]
    x = scores.reshape(y.shape)
    r = scipy.stats.pearsonr(x,y)
    rvalues.append(r[0])
    pvalues.append(r[1])
    names.append(name)

isort = np.argsort(rvalues)[::-1]
for i in isort:
    print '==> {}: {} {}'.format(names[i], rvalues[i], pvalues[i])


for i, name in enumerate(i_header):
    print '==> ' + name         
    y = testX[:,i]
    x = scores.reshape(y.shape)

    h2d, binx, biny = np.histogram2d(x, y, bins=(20, 20))
    h2d /= np.sum(h2d, axis=1)[:, np.newaxis]

    ext = [0, biny[-1], 0, 1]
    plt.imshow(h2d, origin='lower', interpolation='none', cmap='Blues', aspect='auto', extent=ext)
    plt.ylabel('NN score')
    plt.xlabel(name)
    plt.colorbar().set_label('Pr({} | NN score)'.format(name))
    plt.savefig('h2d_{}.png'.format(name))
    plt.close()


m_header = dset['header/metadata']

sort_key = np.argsort(testY)
testM_sorted = testM[sort_key]
scores_sorted = scores[sort_key]

isig = np.where(scores_sorted >= 0.5)[0]
ibkg = np.where(scores_sorted < 0.5)[0]


for i, name in enumerate(m_header):
    print '==> ' + name         
    y = testM[:,i]
    x = scores.reshape(y.shape)

    h2d, binx, biny = np.histogram2d(x, y, bins=(20, 20))
   
    h2d /= np.sum(h2d, axis=1)[:, np.newaxis]

    ext = [0, biny[-1], 0, 1]
    plt.imshow(h2d, origin='lower', interpolation='none', cmap='Blues', aspect='auto', extent=ext)
    plt.ylabel('NN score')
    plt.xlabel(name)
    plt.colorbar().set_label('Pr({} | NN score)'.format(name))
    plt.savefig('h2d_{}.png'.format(name))
    plt.close()

        



for i, name in enumerate(m_header):
    print '==> ' + name


    xsig = testM_sorted[isig, i]
    xbkg = testM_sorted[ibkg, i] 

    plt.hist(xsig, normed=True, histtype='step', label='sig')
    plt.hist(xbkg, normed=True, histtype='step', label='bkg')
    plt.legend(loc='best')
    plt.savefig('h1d_{}.png'.format(name))
    plt.close()

