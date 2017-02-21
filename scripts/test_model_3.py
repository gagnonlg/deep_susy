import numpy as np
import model
import utils

NTRAIN=1000

np.random.seed(900297)
reference_auc = 9.986175613782712901e-01# for above seed

definition = model.ModelDefinition(
    name='TestModel',
    n_hidden_layers=1,
    n_hidden_units=100,
    learning_rate=0.01,
    momentum=0.5,
    l2_reg=1e-6,
    min_epochs=10,
    max_epochs=1000,
    patience=10,
    reweight=True,
    normalize=True,
    early_stop_metric='fmeasure'
)

# ####################################

data_X = np.random.normal([0,0], [1,1], size=[NTRAIN,2])
data_Y = np.logical_xor(
    data_X[:,0] > 0,
    data_X[:,1] > 0
).astype(np.float32)

test_X = np.random.normal([0,0], [1,1], size=[1000,2])
test_Y = np.logical_xor(
    test_X[:,0] > 0,
    test_X[:,1] > 0
).astype(np.float32)

# #####################################

def main():
    definition.log()
    definition.save()
    trained = definition.train(data_X, data_Y)
    trained.save()
    metrics = trained.evaluate(test_X, test_Y)
    metrics.save()

    # np.savetxt('reference_AUC', np.array([metrics.auc]))

    if np.isclose(metrics.auc['unweighted'], reference_auc):
        print '* AUC match :) *'
    else:
        raise RuntimeError('AUC mismatch')

utils.main(main, 'test_model')
