""" Neural networks model definitions and training and testing functions """

# pylint: disable=invalid-name,too-many-arguments,too-many-instance-attributes
# pylint: disable=too-few-public-methods,no-member

import logging
import sys
import tempfile

import h5py as h5
import keras
import numpy as np
import sklearn.metrics
from ROOT import RooStats

import metrics
import utils


class ModelDefinition(object):
    """ Model definition which can be trained """

    @staticmethod
    def from_file(path):
        """ Create a ModelDefinition from a saved file """
        definition = {}
        with open(path, 'r') as deff:
            for line in deff:
                fields = line.strip().split()
                if len(fields) == 2:
                    key, value = fields
                    if key in ['reweight', 'normalize']:
                        value = value == 'True'
                    elif key not in ['name', 'early_stop_metric']:
                        value = float(value)
                    definition[key] = value
        return ModelDefinition(**definition)

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
                 normalize,
                 reduceLR_factor=1,
                 reduceLR_patience=5,
                 early_stop_metric='loss'):
        self.name = name
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
        self.reduceLR_factor = reduceLR_factor
        self.reduceLR_patience = reduceLR_patience
        self.early_stop_metric = early_stop_metric

        self.logger = logging.getLogger('ModelDefinition:' + name)

    def log(self):
        """ Dump the definition to the module log """
        self.logger.info('Dumping ' + self.name + ' definition:')
        for key, value in self.__iter_definition():
            self.logger.info('%s: %s', key, value)

    def save(self):
        """ Save the definition to path = self.name + definition.txt """
        path = utils.unique_path(self.name + '.definition.txt')

        with open(path, 'w') as savefile:
            for key, value in self.__iter_definition():
                savefile.write('{} {}\n'.format(key, value))

        self.logger.info('Model definition saved to %s', path)

    def __build_model(self, shape):
        self.logger.info('Building the keras model')
        model = build_model(
            n_in=shape,
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
            loss='binary_crossentropy',
            metrics=['precision', 'recall', 'fmeasure']
        )
        return model

    def __calc_normalization(self, data_X):
        if self.normalize:
            self.logger.info('Computing the normalization constants')
            return {
                'mean': np.mean(data_X, axis=0),
                'std': np.std(data_X, axis=0)
            }
        return {
            'mean': 0,
            'std': 1,
        }

    def __calc_weight_dict(self, data_Y):
        weightd = None
        if self.reweight:
            self.logger.info('Computing the reweighting constants')
            total = float(data_Y.shape[0])
            total_1 = np.count_nonzero(data_Y)

            w0 = total / (total - total_1)
            w1 = total / total_1

            # implement it this way such that the learning rate would
            # be unnafected in the 50/50 regime
            weightd = {
                0: w0 / max(w0, w1),
                1: w1 / max(w0, w1)
            }
            self.logger.info(
                'Class weights: positive=%f, negative=%f',
                weightd[1],
                weightd[0]
            )
        return weightd

    def __callbacks(self):
        checkpoint_path = tempfile.NamedTemporaryFile()

        metric = 'val_{}'.format(self.early_stop_metric)

        if metric == 'val_loss':
            mode = 'min'
        else:  # precision, recall, fmeasure
            mode = 'max'

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                checkpoint_path.name,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                monitor=metric,
                mode=mode
            ),
            keras.callbacks.EarlyStopping(
                patience=self.patience,
                verbose=1,
                monitor=metric,
                mode=mode
            )
        ]

        if self.reduceLR_factor < 1:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    factor=self.reduceLR_factor,
                    patience=self.reduceLR_patience,
                    verbose=1,
                    monitor=metric,
                    mode=mode
                )
            )

        return callbacks, checkpoint_path

    def train(self, data_X, data_Y):
        """ Produce a trained model from this definition, given some data """

        model = self.__build_model(data_X.shape[1])
        normalization = self.__calc_normalization(data_X)
        weightd = self.__calc_weight_dict(data_Y)
        callbacks, checkpoint = self.__callbacks()

        self.logger.info('Training the model')

        klog = utils.LoggerWriter(self.logger, logging.INFO)
        sys.stdout = klog

        history = model.fit(
            x=(data_X - normalization['mean'])/normalization['std'],
            y=data_Y,
            nb_epoch=self.max_epochs,
            verbose=2,
            callbacks=callbacks,
            validation_split=0.1,
            class_weight=weightd,
        )

        model.load_weights(checkpoint.name)
        checkpoint.close()

        self.logger.info('Done training the model')

        return TrainedModel(
            definition=self,
            internal_model=model,
            normalization=normalization,
            fit_history=history.history
        )

    def __iter_definition(self):
        members = [attr for attr in vars(self) if not attr.startswith("__")]
        for attr in sorted(members):
            if attr != 'logger':
                yield attr, getattr(self, attr)


def build_model(n_in, n_hlayer, n_hunits, l2):
    """ build a keras model

    Arguments:
      n_in: number of input units
      n_hlayer: number of hidden layers
      n_hunits: number of hidden units per hidden layer
      l2: l2 norm regularizer
    Returns:
      A keras model
    """
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

class TrainedModel(object):
    """ Trained model which can be tested and yield predictions """

    def __init__(self,
                 definition,
                 internal_model,
                 normalization,
                 fit_history=None):
        self.definition = definition
        self.internal_model = internal_model
        self.normalization = normalization
        self.fit_history = fit_history

        if self.normalization is None:
            self.normalization = {'mean': 0, 'std': 1}

        self.logger = logging.getLogger('TrainedModel:' + self.definition.name)

    def save(self):
        """ Save the trained model """

        # Save keras model
        path = utils.unique_path(self.definition.name + '.keras_model.h5')
        self.internal_model.save(path)
        self.logger.info('Keras model saved to %s', path)

        # Save normalization constants
        try:
            norm = np.empty((2, self.normalization['mean'].shape[0]))
        except AttributeError:
            norm = np.empty(2)
        norm[0] = self.normalization['mean']
        norm[1] = self.normalization['std']
        npath = utils.unique_path(self.definition.name + '.normalization.txt')
        np.savetxt(npath, norm)
        self.logger.info('Normalization constants saved to %s', npath)

        # Save fit history
        if self.fit_history is not None:
            hpath = utils.unique_path(
                self.definition.name + '.fit_history.txt'
            )
            hfile = h5.File(hpath, 'x')
            for key, val in self.fit_history.iteritems():
                hfile.create_dataset(key, data=val)
            hfile.close()
            self.logger.info('Fit history saved to %s', hpath)
        else:
            self.logger.warning('TrainedModel has no fit history')

    @staticmethod
    def from_files(definition_path, keras_path, norm_path):
        """ Create trained model from saved files.

        Arguments:
          definition_path: path to saved ModelDefinition
          keras_path: path to saved keras model
          norm_path: path to saved normalization
        Returns:
          TrainedModel object
        """
        definition = ModelDefinition.from_file(definition_path)
        internal_model = keras.models.load_model(keras_path)
        np_norm = np.loadtxt(norm_path)
        normalization = {'mean': np_norm[0], 'std': np_norm[1]}
        return TrainedModel(definition, internal_model, normalization)

    def predict(self, data_X):
        """ Yield predictions for given data """
        return self.internal_model.predict(
            (data_X - self.normalization['mean'])/self.normalization['std']
        )[:, 0]

    def evaluate(self, data_X, data_Y, weights=None, lumi=35):
        """ Compute Metrics for this model """
        if weights is None:
            weights = np.ones_like(data_Y)

        self.logger.info('Computing all metrics')

        scores = self.predict(data_X)

        # ROC curves
        roc = {
            'unweighted': sklearn.metrics.roc_curve(data_Y, scores),
            'weighted': sklearn.metrics.roc_curve(
                data_Y,
                scores,
                sample_weight=weights,
                drop_intermediate=False
            )
        }

        pred_class = scores > 0.5

        auc = {
            'unweighted': sklearn.metrics.auc(
                roc['unweighted'][0],
                roc['unweighted'][1]
            ),
            'weighted': sklearn.metrics.auc(
                roc['weighted'][0],
                roc['weighted'][1],
                reorder=True
            )
        }

        accuracy = {
            'unweighted': sklearn.metrics.accuracy_score(data_Y, pred_class),
            'weighted': sklearn.metrics.accuracy_score(
                data_Y,
                pred_class,
                sample_weight=weights
            )
        }

        fmeasure = {
            'unweighted': sklearn.metrics.f1_score(data_Y, pred_class),
            'weighted': sklearn.metrics.f1_score(
                data_Y,
                pred_class,
                sample_weight=weights
            )
        }

        precision = {
            'unweighted': sklearn.metrics.precision_score(data_Y, pred_class),
            'weighted': sklearn.metrics.precision_score(
                data_Y,
                pred_class,
                sample_weight=weights
            )
        }

        recall = {
            'unweighted': sklearn.metrics.recall_score(data_Y, pred_class),
            'weighted': sklearn.metrics.recall_score(
                data_Y,
                pred_class,
                sample_weight=weights
            )
        }

        # significance_at_min_background_efficiency
        sign = calc_significance(
            data_Y,
            scores,
            weights,
            roc['weighted'],
            lumi
        )

        return Metrics(
            name=self.definition.name,
            scores_pos=scores[np.where(data_Y == 1)],
            scores_neg=scores[np.where(data_Y == 0)],
            roc=roc,
            auc=auc,
            accuracy=accuracy,
            fmeasure=fmeasure,
            precision=precision,
            recall=recall,
            significance=sign
        )

def calc_significance(data_Y, scores, weights, roc, lumi):
    """ compute significance at min. bkg efficiency (where stat uncert = 0.3 """
    btotsel = weights[np.where(data_Y == 0)]
    btot = np.sum(btotsel)
    btot2 = np.sum(btotsel*btotsel)
    stot = np.sum(weights[np.where(data_Y == 1)])
    roc2 = sklearn.metrics.roc_curve(
        data_Y,
        scores,
        sample_weight=(weights*weights),
        drop_intermediate=False
    )

    uncert = np.sqrt(roc2[0] * btot2) / (roc[0] * btot)
    ifirstb = np.where(uncert > 0.3)[0][-1]

    fpr_best = roc[0][ifirstb]
    tpr_best = roc[1][ifirstb]

    return RooStats.NumberCountingUtils.BinomialExpZ(
        lumi * tpr_best * stot,
        lumi * fpr_best * btot,
        0.3
    )


class Metrics(object):
    """ Collection of metrics """

    def __init__(
            self,
            name,
            scores_pos,
            scores_neg,
            roc,
            auc,
            accuracy,
            fmeasure,
            precision,
            recall,
            significance
    ):
        self.name = name
        self.scores_pos = scores_pos
        self.scores_neg = scores_neg
        self.roc = roc
        self.auc = auc
        self.accuracy = accuracy
        self.fmeasure = fmeasure
        self.precision = precision
        self.recall = recall
        self.significance = significance

        self.logger = logging.getLogger('Metrics:' + self.name)

    def save(self):
        """ Save the metrics to an hdf5 file """

        path = utils.unique_path(self.name + '.metrics.h5')
        savefile = h5.File(path, 'x')

        savefile.create_dataset('scores/positive', data=self.scores_pos)
        savefile.create_dataset('scores/negative', data=self.scores_neg)

        savefile.create_dataset(
            'roc/unweighted/fp',
            data=self.roc['unweighted'][0]
        )
        savefile.create_dataset(
            'roc/unweighted/tp',
            data=self.roc['unweighted'][1]
        )
        savefile.create_dataset(
            'roc/weighted/fp',
            data=self.roc['weighted'][0]
        )
        savefile.create_dataset(
            'roc/weighted/tp',
            data=self.roc['weighted'][1]
        )

        for metric in ['auc', 'accuracy', 'fmeasure', 'precision', 'recall']:
            savefile.create_dataset(
                '{}/unweighted'.format(metric),
                data=getattr(self, metric)['unweighted']
            )
            savefile.create_dataset(
                '{}/weighted'.format(metric),
                data=getattr(self, metric)['weighted']
            )

        savefile.create_dataset('significance', data=self.significance)

        savefile.close()

        self.logger.info('Metrics saved to %s', path)
