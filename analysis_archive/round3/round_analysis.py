import itertools
import sqlite3

import numpy as np
from matplotlib import pyplot as plt

#db = sqlite3.connect('total.db')
db = sqlite3.connect('round_3_fixed.db')

cur = db.execute('select * from perf')
names = [description[0] for description in cur.description]
names.remove('ID')
names.remove('AUC')
names.remove('min_epochs')
names.remove('max_epochs')
names.remove('REWEIGHT')

# print names

names.append('n_hidden_units*n_hidden_layers')

# for col in names:
#     print '=> ' + col
#     result = list(db.execute('select {},AUC from perf'.format(col)))
#     values = np.array([val for val,auc in result]).astype('float32')
#     aucs = np.array([auc for val, auc in result]).astype('float32')
#     plt.scatter(values, aucs)
#     plt.xlabel(col)
#     plt.ylabel('AUC')
#     if col == 'l2_reg':
#         plt.axis([0, 1e-4, 0.98, 1])
#     if col == 'n_hidden_units*n_hidden_layers':
#         col = 'n_hidden_units_total'
#     plt.savefig('{}_vs_auc.png'.format(col))
#     plt.close()


    


# for (c1, c2) in itertools.combinations(names, 2):
#     print '=> ' + c1 + ',' + c2
#     result = list(db.execute('select {},{},AUC from perf'.format(c1,c2)))

#     v1s =  [v1 for v1,v2,auc in result]
#     v2s =  [v2 for v1,v2,auc in result]
#     aucs =  np.array([auc for v1,v2,auc in result])

#     aucs -= np.min(aucs)
#     aucs /= np.max(aucs)
#     #print aucs


#     plt.scatter(v1s, v2s, s=20*4**(2*aucs))
#     #plt.colorbar()
#     plt.xlabel(c1)
#     plt.ylabel(c2)

#     if c1 == 'n_hidden_units*n_hidden_layers':
#         c1 = 'n_hidden_units_total'
#     if c2 == 'n_hidden_units*n_hidden_layers':
#         c2 = 'n_hidden_units_total'

    
#     plt.savefig('{}_{}_auc.png'.format(c1,c2))
#     plt.close()

result0 = [v for v, in db.execute('select AUC from perf where normalize=0')]
result1 = [v for v, in db.execute('select AUC from perf where normalize=1')]

plt.hist(result0, label='normalize=0', histtype='step', normed=True, bins=20)
plt.hist(result1, label='normalize=1', histtype='step', normed=True, bins=20)
plt.legend(loc='upper left')
plt.savefig('AUC_normalize_0_1.png')
plt.close()
