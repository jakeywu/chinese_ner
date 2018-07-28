import warnings
import os
from sklearn import metrics
import tensorflow as tf
warnings.filterwarnings('ignore')


class BaseModel(object):
    checkpointPath = "checkpoints/"

    def __init__(self):
        self.sess = tf.Session()
        self.template = self.__template()

    @staticmethod
    def __template():
        return """<<Train>> EPOCH: [%d] STEP: [%d] LOSS: [%.3f] \t ACC: [%.3f] \t RECALL: [%.3f] \t F1: [%.3f]"""

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
        self.embedding_size = conf.embedding_size
        self.vocab_size = conf.vocab_size
        self.num_hidden = conf.num_hidden
        self.num_tag = conf.num_tag
        self.epoch = conf.epoch
        self.filter_size = conf.filter_size
        self.filter_num = conf.filter_num
        self.learning_rate = conf.learning_rate

        self._init_placeholder()
        self._embedding_layers()
        self._bi_lstm_layers()
        self._build_train_op()

    def _init_placeholder(self):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="inputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="targets")
        self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
        self.sequence_len = tf.reduce_sum(
            tf.cast(tf.not_equal(tf.cast(-1, self.inputs.dtype), self.inputs), tf.int32), axis=1
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
            outputs = tf.nn.dropout(tf.concat([output_fw, output_bw], axis=2), keep_prob=self.keep_prob)
            shape = tf.shape(outputs)
            bi_output = tf.reshape(outputs, [-1, 2 * self.num_hidden])
            lstm_w = tf.get_variable(
                name="W_out", shape=[2 * self.num_hidden, self.num_tag], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            lstm_b = tf.get_variable(name="b", shape=[self.num_tag], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))
            logit = tf.matmul(bi_output, lstm_w) + lstm_b
            self.logits = tf.reshape(logit, [-1, shape[1], self.num_tag])

    def _build_train_op(self):
        # log-likelihood and transition matrix
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            inputs=self.logits, tag_indices=self.targets, sequence_lengths=self.sequence_len)
        self.loss = tf.reduce_mean(-log_likelihood)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    @staticmethod
    def __viterbi_decode_metric(logits, labels, seq_len, transition_params):
        y_pred = []
        y_true = []
        for i in range(len(seq_len)):
            score = logits[i][0:seq_len[i]]
            viterbi, _ = tf.contrib.crf.viterbi_decode(score=score, transition_params=transition_params)
            y_pred.extend(viterbi)
            y_true.extend(labels[i][0:seq_len[i]])

        accuracy = metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro")
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, average="macro")
        f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred, average="macro")

        return accuracy, recall, f1

    def train(self, train):
        self.sess.run(tf.global_variables_initializer())
        step = 0
        for i in range(self.epoch):
            try:
                while True:
                    train_x, train_y = next(train)
                    step += (i + 1) * len(train_x)
                    loss, seq_len, logits, trans_params, _ = self.sess.run(
                        fetches=[self.loss, self.sequence_len, self.logits, self.transition_params, self.train_op],
                        feed_dict={self.inputs: train_x, self.targets: train_y, self.keep_prob: 0.5})
                    accuracy, recall, f1 = self.__viterbi_decode_metric(
                        logits=logits, labels=train_y, seq_len=seq_len, transition_params=trans_params)
                    print(self.template % (i+1, step, loss, accuracy, recall, f1))
            except StopIteration as e:
                continue

    def _cnn_layers(self):
        with tf.variable_scope(name_or_scope="cnn_layers"):
            self.embedded_inputs_expanded = tf.expand_dims(self.embedded_inputs, -1)

            conv1 = self._cnn_2d(
                inputs=self.embedded_inputs_expanded, scope_name="conv", filter_height=self.filter_size,
                filter_width=self.embedding_size, in_channels=1, out_channel=self.filter_num
            )
            conv1 = tf.nn.relu(conv1)
            conv1 = self._cnn_max_pool(inputs=conv1, scope_name="max_pool", ksize=self.sequence_len-self.filter_size + 1)
