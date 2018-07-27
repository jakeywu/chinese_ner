import os
import tensorflow as tf


class BaseModel(object):
    checkpointPath = "checkpoints/"

    def __init__(self):
        self.sess = tf.Session()

    @staticmethod
    def __exists_checkpoint():
        os.makedirs(BaseModel.checkpointPath)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "{}short-name".format(BaseModel.checkpointPath))

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(BaseModel.checkpointPath))

    @staticmethod
    def _cnn_2d(inputs, scope_name, filter_height, filter_width, in_channels, out_channel):
        with tf.variable_scope(name_or_scope=scope_name):
            filters = tf.get_variable(
                name="W", shape=[filter_height, filter_width, in_channels, out_channel], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01))
            bias = tf.get_variable(name="b", shape=[out_channel], dtype=tf.float32, initializer=tf.constant_initializer())
            con2d_op = tf.nn.conv2d(input=inputs, filter=filters, strides=[1, 1, 1, 1], padding="VALID")
        return tf.nn.bias_add(value=con2d_op, bias=bias)

    @staticmethod
    def _cnn_max_pool(inputs, scope_name, ksize):
        with tf.variable_scope(name_or_scope=scope_name):
            return tf.nn.max_pool(value=inputs, ksize=[1, ksize, 1, 1], strides=[1, 1, 1, 1], padding="VALID")


class RnnCnnCrf(BaseModel):
    def __init__(self, conf):
        super(RnnCnnCrf, self).__init__()
        self.epoch = conf.epoch
        self.embedding_size = conf.embedding_size
        self.vocab_size = conf.vocab_size
        self.num_hidden = conf.num_hidden
        self.num_tag = conf.num_tag
        self.filter_size = conf.filter_size
        self.filter_num = conf.filter_num

        self._init_placeholder()
        self._embedding_layers()
        self._bi_lstm_layers()
        self._cnn_layers()

    def _init_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="targets")
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=None, name="batch_size")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self.sequence_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(-1, self.inputs.dtype), self.inputs), tf.int32), axis=0
        )

    def _embedding_layers(self):
        with tf.variable_scope(name_or_scope="embedding_layer"):
            embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=[self.vocab_size, self.embedding_size], dtype=tf.float32)
            self.embedded_inputs = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.inputs)

    def _bi_lstm_layers(self):
        with tf.variable_scope(name_or_scope="biLSTM_layers"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.num_hidden)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.num_hidden)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs=self.embedded_inputs, sequence_length=self.sequence_len,
                time_major=False, dtype=tf.float32)
            self.lstm_ouputs = tf.nn.dropout(tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)

    def _cnn_layers(self):
        with tf.variable_scope(name_or_scope="cnn_layers"):
            self.embedded_inputs_expanded = tf.expand_dims(self.embedded_inputs, -1)
            conv1 = self._cnn_2d(
                inputs=self.embedded_inputs_expanded, scope_name="conv", filter_height=self.filter_size,
                filter_width=self.embedding_size, in_channels=1, out_channel=self.filter_num
            )
            conv1 = tf.nn.relu(conv1)
            conv1 = self._cnn_max_pool(inputs=conv1, scope_name="max_pool", ksize=self.sequence_len-self.filter_size + 1)
            print("="*10)
            print(conv1)
