import h5py
import keras
import numpy as np
import subprocess

print("=> loading data")

input_file = h5py.File('gtt_deep_learning_dataset_0.h5', 'r')
x_train = input_file['training']['inputs']
y_train = input_file['training']['labels']

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
std[np.where(std == 0)] = 1

norm = np.zeros((2,std.shape[0]))
norm[0] = mean
norm[1] = std
np.savetxt('norm.txt', norm)

x_train -= mean
x_train /= std


print("=> generating model")

id = subprocess.check_output(['uuidgen']).split('-')[0]
print "id={}".format(id)


n_hidden_layers = np.random.randint(2,4)
n_hidden_units = np.random.randint(100,800)

print "n_hidden_layers={}".format(n_hidden_layers)
print "n_hidden_units={}".format(n_hidden_units)

reg = np.random.choice([None, keras.regularizers.l2, keras.regularizers.l1])
print "reg={}".format(str(reg))

if reg is None:
    reg = lambda _: None

reg_w = np.random.uniform(1e-7,1e-3)
if reg is not None:
    print "reg_w={}".format(reg_w)

lr = np.random.uniform(0.001, 0.1)
print "lr={}".format(lr)

reduceLR_patience = np.random.randint(1,25)
print "reduceLR_patience={}".format(reduceLR_patience)

reduceLR_factor = np.random.uniform(0.1, 0.9)
print "reduceLR_factor={}".format(reduceLR_factor)


print("=> building model")


model = keras.models.Sequential()

structure = [x_train.shape[1]] + ([n_hidden_units] * n_hidden_layers) + [1]

for i in range(1,len(structure)):

    model.add(
        keras.layers.Dense(
            input_dim=structure[i-1],
            output_dim=structure[i],
            init='glorot_uniform',
            W_regularizer=reg(reg_w)
        )
    )

    if i < (len(structure) - 1):
        model.add(keras.layers.Activation('relu'))

    else:
        model.add(keras.layers.Activation('sigmoid'))
            

model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.SGD(lr=lr),
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "fit_model_{}.hdf5".format(id),
        verbose=1,
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        min_delta=0.0001,
        patience=50,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=reduceLR_factor,
        patience=reduceLR_patience,
        verbose=1,
    )
]

print("=> fitting model")

model.fit(
    x=x_train,
    y=y_train,
    nb_epoch=1000,
    verbose=2,
    callbacks=callbacks,
    validation_split=0.1,
    shuffle=True
)
    


    
