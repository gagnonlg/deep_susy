import os
import logging
import sys
import tempfile

import h5py as h5
import keras
import numpy as np

import metrics
import utils

class ModelDefinition(object):
    def __init__(self,
                 name,
                 n_hidden_layers,
                 n_hidden_units,
                 learning_rate,
                 momentum,
                 l2_reg,
                 min_epochs,
                 max_epochs,
                 patience,
                 reweight,
                 normalize):
        self.name=name
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.patience = patience
        self.reweight = reweight
        self.normalize = normalize

        self.logger = logging.getLogger('ModelDefinition:' + name)


    def log(self):
        self.logger.info('Dumping ' + self.name + ' definition:')
        for key, value in self.__iter_definition():
            self.logger.info('%s: %s', key, value)

            
    def save(self):

        path = utils.unique_path(self.name + '.definition.txt')
 
        with open(path, 'w') as savefile:
            for key, value in self.__iter_definition():
                savefile.write('{} {}\n'.format(key, value))

        self.logger.info('Model definition saved to %s', path)


    def train(self, data_X, data_Y):

        self.logger.info('Building the keras model')
        model = build_model(
            n_in=data_X.shape[1],
            n_hlayer=self.n_hidden_layers,
            n_hunits=self.n_hidden_units,
            l2=self.l2_reg
        )
        
        self.logger.info('Compiling the model')
        model.compile(
            optimizer=keras.optimizers.SGD(
                lr=self.learning_rate,
                momentum=self.momentum,
            ),
            loss='binary_crossentropy'
        )

        normalization = None
        if self.normalize:
            self.logger.info('Computing the normalization constants')
            normalization = {
                'mean': np.mean(data_X, axis=0),
                'std': np.std(data_X, axis=0)
            }
        else:
            normalization = {
                'mean': 0,
                'std': 1,
            }

        weightd = None
        if self.reweight:
            self.logger.info('Computing the reweighting constants')
            total = float(data_Y.shape[0])
            total_1 = np.count_nonzero(data_Y)
            weightd = {
                0: total / (total - total_1),
                1: total / total_1
            }
            self.logger.info('Class weights:')
            self.logger.info(
                'positive: %f, negative: %f',
                weightd[1],
                weightd[0]
            )
            
        checkpoint_path = tempfile.NamedTemporaryFile()

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                checkpoint_path.name,
                verbose=1,
                save_best_only=True,
                save_weights_only=True
            ),
            keras.callbacks.EarlyStopping(
                patience=self.patience,
                verbose=1
            )
        ]

        self.logger.info('Training the model')

        klog = utils.LoggerWriter(self.logger, logging.INFO)
        sys.stdout = klog
        
        model.fit(
            x=(data_X - normalization['mean'])/normalization['std'],
            y=data_Y,
            nb_epoch=self.max_epochs,
            verbose=2,
            callbacks=callbacks,
            validation_split=0.1,
            class_weight=weightd,
        )

        model.load_weights(checkpoint_path.name)
        checkpoint_path.close()

        self.logger.info('Done training the model')
                
        return TrainedModel(
            definition=self,
            internal_model=model,
            normalization=normalization
        )

    
    def __iter_definition(self):
        members = [attr for attr in vars(self) if not attr.startswith("__")]
        for attr in sorted(members):
            if attr != 'logger':
                yield attr, getattr(self, attr)


def build_model(n_in, n_hlayer, n_hunits, l2):

    model = keras.models.Sequential()

    struct = [n_in] + ([n_hunits]*n_hlayer) + [1]

    acts = ['relu']*len(struct)
    acts[-1] = 'sigmoid'

    for i in range(1, len(struct)):
        model.add(
            keras.layers.Dense(
                input_dim=struct[i-1],
                output_dim=struct[i],
                activation=acts[i],
                W_regularizer=keras.regularizers.l2(l2),
                init='glorot_uniform'
            )
        )

    return model


########################################################################
#

class TrainedModel(object):

    def __init__(self, definition, internal_model, normalization):
        self.definition = definition
        self.internal_model = internal_model
        self.normalization = normalization

        if self.normalization is None:
            self.normalization = {'mean': 0, 'std': 1}

        self.logger = logging.getLogger('TrainedModel:' + self.definition.name)

    def save(self):

        path = utils.unique_path(self.definition.name + '.keras_model.h5')
        self.internal_model.save(path)

        self.logger.info('Keras model saved to %s', path)

        norm = np.empty((2,self.normalization['mean'].shape[0]))
        norm[0] = self.normalization['mean']
        norm[1] = self.normalization['std']
        npath = utils.unique_path(self.definition.name + '.normalization.txt')
        np.savetxt(npath, norm)

        self.logger.info('Normalization constants saved to %s', npath)

    def predict(self, data_X):
        return self.internal_model.predict(
            (data_X - self.normalization['mean'])/self.normalization['std']
        )[:,0]

    def evaluate(self, data_X, data_Y, weights=None):

        if weights is None:
            weights = np.ones_like(data_Y)

        self.logger.info('Computing prediction scores')
        scores = self.predict(data_X)

        self.logger.info('Computing the ROC curve and AUC')
        roc = metrics.roc_curve(scores, data_Y)
        auc = metrics.auc(roc)
        self.logger.info('AUC: %f', auc)

        return Metrics(
            name=self.definition.name,
            scores_pos=scores[np.where(data_Y == 1)],
            scores_neg=scores[np.where(data_Y == 0)],
            roc=roc,
            auc=auc
        )


class Metrics(object):

    def __init__(self, name, scores_pos, scores_neg, roc, auc):
        self.name = name
        self.scores_pos = scores_pos
        self.scores_neg = scores_neg
        self.roc = roc
        self.auc = auc

        self.logger = logging.getLogger('Metrics:' + self.name)

    def save(self):
    
        path = utils.unique_path(self.name + '.metrics.h5')
        savefile = h5.File(path, 'x')
        savefile.create_dataset('roc/fp', data=self.roc[0])
        savefile.create_dataset('roc/tp', data=self.roc[1])
        savefile.create_dataset('scores/positive', data=self.scores_pos)            
        savefile.create_dataset('scores/negative', data=self.scores_neg)   
        savefile.create_dataset('auc', data=self.auc)
        savefile.close()

        self.logger.info('Metrics saved to %s', path)

