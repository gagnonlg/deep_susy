class ConvolutionLayer(object):

    def __init__(self, filter_shape, pool_size):
        """ Convolution + MaxPooling layer 

        Arguments:
          filter_shape: tuple (n_filters, n_features, xdim, ydim)
        """
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])) \
                   // np.prod(pool_size)

        self.filter_shape = filter_shape
        self.pool_size = pool_size

        W_bound = 1.0 / np.sqrt(6.0 / (fan_in + fan_out))
        self.W = theano.shared(
            np.random.uniform(
                low=-W_bound,
                high=W_bound,
                size=filter_shape
            ).astype(theano.config.floatX)
        )

        self.b = theano.shared(
            np.zeros((filter_shape[0],)).astype(theano.config.floatX)
        )

        self.params = [self.W, self.b]

    def expression(self, X, input_shape=None):
        
        conv = theano.tensor.nnet.conv2d(
            input=X,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=input_shape
        )

        poold = theano.signal.pool.pool_2d(
            input=conv,
            ds=self.pool_size,
            ignore_border=True
        )

        output = T.tanh(poold + self.b.dimshuffle('x', 0, 'x', 'x'))

        return output
            
