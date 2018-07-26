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


class RnnCnnCrf(object):
    def __init__(self, conf):
        self.epoch = conf.epoch
        self.embedding_size = conf.embedding_size
        self.vocab_size = conf.vocab_size
        self.num_hidden = conf.num_hidden
        self.num_tag = conf.num_tag

        self._init_placeholder()
        self._embedding_layers()
        self._bi_lstm_layers()

    def _init_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="targets")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self.sequence_len = tf.reduce_sum(
            tf.cast(tf.not_equal(99999999, self.inputs), tf.int32), axis=0
        )
        self.batch_size = tf.shape(self.inputs)[0]

    def _embedding_layers(self):
        with tf.variable_scope(name_or_scope="embedding_layer"):
            embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=[self.vocab_size, self.embedding_size], dtype=tf.float32)
            self.embedded_inputs = tf.nn.embedding_lookup(params=embedding_matrix, ids=self.inputs)

    def _bi_lstm_layers(self):
        with tf.variable_scope(name_or_scope="biLSTM_layers"):
            shape = tf.shape(self.embedded_inputs)
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.num_hidden)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.num_hidden)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs=self.embedded_inputs, sequence_length=self.sequence_len,
                time_major=False, dtype=tf.float32)
            outputs = tf.nn.dropout(tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)
            print(outputs)
            # bi_outputs = tf.reshape(outputs, [-1, 2 * self.num_hidden])
            # with tf.variable_scope(name_or_scope="proj"):
            #     weight = tf.get_variable(name="W_out", shape=[2 * self.num_hidden, self.num_tag], dtype=tf.float32,
            #                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            #     b = tf.get_variable(name="b", shape=[self.num_tag], dtype=tf.float32, initializer=tf.zeros_initializer())
            #     pred = tf.matmul(bi_outputs, weight) + b
            #     self.lstm_logits = tf.reshape(pred, [-1, shape[1], self.num_tag])

    def _cnn_layers(self):
        pass
