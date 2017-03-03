"""Generic LSTM language model."""
from emLam.nn.rnn import get_rnn

import tensorflow as tf

class LSTMModel(object):
    """Generic LSTM language model based on the PTB model in tf/models."""
    def __init__(self, params, is_training, softmax, need_prediction=False):
        self.is_training = is_training
        self.params = params

        self._data()
        outputs = self._build_network()
        if need_prediction:
            self._cost, self.prediction = softmax(
                outputs, self._targets, need_prediction)
        else:
            self._cost = softmax(outputs, self._targets)

        if is_training:
            self._optimize()
        else:
            self._train_op = tf.no_op()

        with tf.name_scope('Summaries') as scope:
            summaries = []
            summaries.append(
                tf.summary.scalar(scope + 'Loss', self._cost))
            if is_training:
                summaries.append(
                    tf.summary.scalar(scope + 'Learning rate',
                                      self._lr))
            self.summaries = tf.summary.merge(summaries)

    def _data(self):
        """
        Creates the input placeholders. If using an embedding, the input is
        a single number per token; if not, it must be one-hot encoded.
        """
        dims = [self.params.batch_size, self.params.num_steps]
        self._input_data = tf.placeholder(
            tf.int32, dims, name='input_placeholder')
        self._targets = tf.placeholder(
            tf.int32, dims, name='target_placeholder')

    def _build_network(self):
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        # D: Not really...
        def _get_layer():
            rnn_cell = get_rnn(self.params.rnn_cell, self.params.hidden_size)
            if self.is_training and self.params.dropout < 1:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(
                    rnn_cell, output_keep_prob=self.params.dropout)
            return rnn_cell
        cell = tf.contrib.rnn.MultiRNNCell(
            [_get_layer() for _ in range(self.params.num_layers)])

        self._initial_state = cell.zero_state(self.params.batch_size,
                                              dtype=self.params.data_type)

        if self.params.embedding:
            # If using an embedding, the input is a single number per token...
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    'embedding', [self.params.vocab_size, self.params.hidden_size],
                    trainable=self.params.embedding_trainable,
                    dtype=self.params.data_type)
                inputs = tf.nn.embedding_lookup(embedding, self._input_data)
        else:
            # ..., if not, it must be one-hot encoded.
            inputs = tf.one_hot(self._input_data, self.params.vocab_size,
                                dtype=self.params.data_type)
            # tf.unpack(inputs, axis=1) only needed for rnn, not dynamic_rnn

        if self.is_training and self.params.dropout < 1:
            inputs = tf.nn.dropout(inputs, self.params.dropout)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

        # outputs = []
        # state = self._initial_state
        # with tf.variable_scope("RNN"):
        #   for time_step in range(num_steps):
        #     if time_step > 0: tf.get_variable_scope().reuse_variables()
        #     (cell_output, state) = cell(inputs[:, time_step, :], state)
        #     outputs.append(cell_output)

        # output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

        # This could be replaced by tf.scan(), see
        # http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
        outputs, state = tf.nn.dynamic_rnn(
            inputs=inputs, cell=cell, dtype=self.params.data_type,
            initial_state=self._initial_state)
        self._final_state = state
        return outputs

    def _optimize(self):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        if self.params.max_grad_norm:
            grads, _ = tf.clip_by_global_norm(grads, self.params.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
