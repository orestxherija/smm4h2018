import keras
from keras import backend as K

# taken from https://github.com/cbaziotis/keras-utilities/blob/master/kutilities/layers.py
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    # todo: check that this is correct
    return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

# taken from https://github.com/cbaziotis/keras-utilities/blob/master/kutilities/layers.py
class AttentionWithContext(keras.layers.Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False, **kwargs):

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = keras.initializers.get('glorot_uniform')

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.u_regularizer = keras.regularizers.get(u_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.u_constraint = keras.constraints.get(u_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            (input_shape[-1], input_shape[-1],),
            initializer=self.init,
            name='{}_W'.format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint)
        
        if self.bias:
            self.b = self.add_weight(
                (input_shape[-1],),
                initializer='zero',
                name='{}_b'.format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint)

        self.u = self.add_weight(
            (input_shape[-1],),
            initializer=self.init,
            name='{}_u'.format(self.name),
            regularizer=self.u_regularizer,
            constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.tanh(dot_product(x, self.W) + (self.b if self.bias else 0))
        ait = dot_product(uit, self.u)
        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        r = K.sum(x * K.expand_dims(a), axis=1)

        if self.return_attention:
            return [r, a]
        return r

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),(input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

def embedding(input_text, max_sequence_length, embedding_matrix, **kwargs):

    trainable = kwargs.get('trainable',False)
    masking = kwargs.get('masking',False)
    gaussian_noise = kwargs.get('gaussian_noise', 0.)
    embedding_do = kwargs.get('embedding_do', 0.)

    vocabulary_sz = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]

    _embedding = keras.layers.Embedding(
        input_dim=vocabulary_sz,
        output_dim=embedding_dim,
        input_length=max_sequence_length if max_sequence_length > 0 else None,
        trainable=trainable,
        mask_zero=masking if max_sequence_length > 0 else False,
        weights=[embedding_matrix]
    )(input_text)
    if gaussian_noise > 0.:
        _embedding = keras.layers.GaussianNoise(gaussian_noise)(_embedding)
    if embedding_do > 0.:
        _embedding = keras.layers.Dropout(embedding_do)(_embedding)

    return _embedding

def rnn_encoder(nb_cells, **kwargs):

    '''
    Create a (bidirectional) RNN layer; can be LSTM or GRU

    Parameters
    ----------
    n_cells: int
        number of hidden units in the RNN

    Optional
    --------
    unit: Keras Layer
        can be set to LSTM or GRU
    bidirectional: bool
        if True, apply Bidirectional wrapper
    dropout_U: float
        fraction of the units to drop for the linear transformation of the inputs
    l2_reg: float
        regularization parameter to apply to the output of the RNN layer


    Returns
    -------
    rnn: Keras Layer
        a recurrent Layer
    '''

    rnn_type = kwargs.get('unit',keras.layers.LSTM)
    bidirectional = kwargs.get('bidirectional',True)
    return_sequences = kwargs.get('return_sequences',True)
    recurrent_do = kwargs.get('recurrent_do',0.)
    l2_reg = kwargs.get('l2_reg',0.)

    rnn = rnn_type(
        units=nb_cells, 
        return_sequences=return_sequences, 
        recurrent_dropout=recurrent_do,
        kernel_regularizer=keras.regularizers.l2(l2_reg))
    
    return keras.layers.Bidirectional(rnn) if bidirectional else rnn

def rnn_encoders_with_attention(nb_cells,embeddings,**kwargs):

    rnn_layers = kwargs.get('rnn_layers',1)
    unit = kwargs.get('unit',keras.layers.LSTM)
    bidirectional = kwargs.get('bidirectional',True)
    l2_reg = kwargs.get('l2_reg',0)
    linear_do = kwargs.get('linear_do', 0)
    recurrent_do = kwargs.get('recurrent_do', 0)
    attention_do = kwargs.get('attention_do', 0)

    # encoding
    for i in range(rnn_layers):
        representation = rnn_encoder(
            nb_cells=nb_cells, 
            rnn_type=unit,
            bidirectional=bidirectional, 
            return_sequences=True,
            recurrent_dropout=recurrent_do,
            kernel_regularizer=keras.regularizers.l2(l2_reg))(embeddings)
        if linear_do > 0:
            representation = keras.layers.Dropout(linear_do)(representation) 
    representation = AttentionWithContext()(representation)
    if attention_do > 0:
        representation = keras.layers.Dropout(attention_do)(representation)

    return representation

def softmax_classifier(representation, nb_classes,**kwargs):

    l2_dense = kwargs.get('l2_dense', 0.)

    # prediction
    if nb_classes > 1:
        activation = 'softmax'
        objective  = 'categorical_crossentropy'
    else:
        activation = 'sigmoid'
        objective = 'binary_crossentropy'

    output_probs = keras.layers.Dense(
        units=nb_classes, 
        activation=activation,
        activity_regularizer=keras.regularizers.l2(l2_dense))(representation)

    return output_probs