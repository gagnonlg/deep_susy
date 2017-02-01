import h5py as h5
import keras
import numpy as np

dataset = h5.File('NNinput-0.h5')

train_X = np.array(dataset['training/inputs'])
train_Y = np.array(dataset['training/labels'])

train_X_mean = np.mean(train_X, axis=0)
train_X_std = np.std(train_X, axis=0)

total = train_Y.shape[0]
total_pos = np.count_nonzero(train_Y)

network = keras.models.Sequential()
network.add(keras.layers.Dense(input_dim=train_X.shape[0], output_dim=300))
network.add(keras.layers.Activation('relu'))
network.add(keras.layers.Dense(output_dim=1))
network.add(keras.layers.Activation('sigmoid'))

network.compile(
    optimizer=keras.optimizers.SGD(lr=0.01),
    loss='binary_crossentropy'
)

checkpt = keras.callbacks.ModelCheckpoint(
    filepath='fitted_model.h5',
    verbose=1,
    save_best_only=True,
)

network.fit(
    x=((train_X - train_X_mean) / train_X_std),
    y=train_Y,
    batch_size=32,
    nb_epochs=100,
    verbose=2,
    callbacks=[checkpt],
    validation_split=0.1,
    class_weight={
        0: float(total) / (total - total_pos),
        1: float(total) / total_pos
    }
)
