""" Custom keras layers """
import keras
import numpy as np


def _config(layer, config):
    base_config = super(layer.__class__, layer).get_config()
    return dict(base_config.items() + config.items())


class ScaleOffset(keras.layers.Layer):
    """ x * scale + offset """

    def __init__(self, scale=1.0, offset=0.0, **kwargs):
        self.scale = scale
        self.offset = offset

        if isinstance(self.scale, dict) and self.scale['type'] == 'ndarray':
            self.scale = np.array(self.scale['value']).astype('float32')

        if isinstance(self.offset, dict) and self.offset['type'] == 'ndarray':
            self.offset = np.array(self.offset['value']).astype('float32')

        super(ScaleOffset, self).__init__(**kwargs)

    def call(self, x):
        return x * self.scale + self.offset

    def get_config(self):
        return _config(self, {
            'scale': self.scale,
            'offset': self.offset
        })
