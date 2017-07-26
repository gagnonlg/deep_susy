import keras

# 10 * 5 + 4 * 4 + 4 * 4 + 2 + 2 =
# 50 + 16 + 16 + 2 + 2 =
# 66 + 18 + 2 =
# 84 + 2 =
# 88

x = keras.layers.Input((88,))

l1 = keras.layers.Dense(
    300,
    activations='tanh',
    kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
    kernel_regularizer=keras.regularizers.l2(1e-5),
)(x)

l2 = keras.layers.Dense(
    300,
    activations='tanh',
    kernel_initializer=keras.initializers.RandomNormal(stddev=0.05),
    kernel_regularizer=keras.regularizers.l2(1e-5),
)(l1)

l3 = keras.layers.Dense(
    300,
    activations='tanh',
    kernel_initializer=keras.initializers.RandomNormal(stddev=0.05),
    kernel_regularizer=keras.regularizers.l2(1e-5),
)(l2)

l4 = keras.layers.Dense(
    300,
    activations='tanh',
    kernel_initializer=keras.initializers.RandomNormal(stddev=0.05),
    kernel_regularizer=keras.regularizers.l2(1e-5),
)(l3)

l5 = keras.layers.Dense(
    300,
    activations='tanh',
    kernel_initializer=keras.initializers.RandomNormal(stddev=0.05),
    kernel_regularizer=keras.regularizers.l2(1e-5),
)(l4)

l6 = keras.layers.Dense(
    5,
    activations='softmax',
    kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
    kernel_regularizer=keras.regularizers.l2(1e-5),
)(l5)

model = keras.models.Model(inputs=x, outputs=l6)

model.compile(
    optimizer=keras.optimizers.SGD(
        lr=0.05
    )
)

class EarlyStopping(keras.callbacks.EarlyStopping):

    def __init__(self, min_epochs, threshold, *args, **kwargs):
        self.min_epochs = min_epochs
        self.threshold
        super(EarlyStopping, self).__init__(*args, **kwargs)
        self.monitor_op = lambda x, y: (y - x) / y > self.threshold

    def on_epoch_end(self, epoch, logs):
        if epoch >= 200:
            return super(EarlyStopping, self).on_epoch_end(epoch, logs)

class MomentumScheduler(keras.callbacks.Callback):

    def __init__(self, start, end, nepochs):
        self.current = start
        self.end = end
        self.nepochs = nepochs
        super(MomentumScheduler, self).__init__()

    def on_epoch_begin(self, epoch, logs):

        if epoch >= end:
            m = end
        else:
            m = epochs * (end - start) / nepochs + start

        keras.backend.set_value(self.model.optimizer.momentum, m)


def schedule_lr(epoch):
    initial = 0.05
    freduce = 1.0000002
    minrate = 1e-6
    minepoch = math.log(initial / minrate, freduce)

    if epoch >= minepoch:
        return 1e-6

    return initial / (freduce ** epoch)

callbacks = [
    keras.callbacks.LearningRateScheduler(schedule_lr),
    MomentumScheduler(
        start=0.9,
        end=0.99,
        nepcohs=200
    ),
    EarlyStopping(
        min_epochs=200,
        threshold=0.00001
    )
]

model.fit(
    batch_size=100,
    callbacks=callbacks
)

