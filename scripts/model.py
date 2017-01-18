import keras
import numpy as np

class ModelDefinition(object):
    def __init__(self,
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


    def train(self, data_X, data_Y):
        
        model = build_model(
            n_in=data_X.shape[1],
            n_hlayer=self.n_hidden_layers,
            n_hunits=self.n_hidden_units,
            l2=self.l2_reg
        )

        model.compile(
            optimizer=keras.optimizers.SGD(
                lr=self.learning_rate,
                momentum=self.momentum,
            ),
            loss='binary_crossentropy'
        )

        normalization = None
        if self.normalize:
            normalization = {
                'mean': np.mean(data_X, axis=0),
                'std': np.std(data_X, axis=0)
            }

        weightd = None
        if self.reweight:
            total = float(data_Y.shape[0])
            total_1 = np.count_nonzero(data_Y)
            weightd = {
                0: total / (total - total_1),
                1: total / total_1
            }

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'fitted_model.h5',
                verbose=1,
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                patience=self.patience,
                verbose=1
            )
        ]

        model.fit(
            x=(data_X - normalization['mean'])/normalization['std'],
            y=data_Y,
            nb_epoch=self.max_epochs,
            verbose=2,
            callbacks=callbacks,
            validation_split=0.1,
            class_weight=weightd,
        )

        model.load_weights('fitted_model.h5')

        return TrainedModel(
            definition=self,
            internal_model=model,
            normalization=normalization
        )


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

