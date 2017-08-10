import keras

m = keras.models.Sequential()

def lmbda(z):
  return keras.layers.Lambda(lambda x: x + z, input_shape=(1,))

m.add(lmbda(100))
m.compile(optimizer='sgd', loss='mse')
m.save('test_bug.h5')

keras.models.load_model('test_bug.h5')
