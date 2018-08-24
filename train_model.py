import tensorflow as tf
from rnn_cnn_crf import RnnCnnCrf

tf.flags.DEFINE_integer(name="num_tag", default=4, help="number tags")
tf.flags.DEFINE_integer(name="epoch", default=2, help="maximum epochs")
tf.flags.DEFINE_integer(name="batch_size", default=10, help="batch size")
tf.flags.DEFINE_integer(name="vocab_size", default=5000, help="vocab num")
tf.flags.DEFINE_integer(name="num_hidden", default=128, help="lstm num hidden")
tf.flags.DEFINE_integer(name="embedding_size", default=128, help="embedding size")
tf.flags.DEFINE_float(name="learning_rate", default=0.01, help="init learning rate")

tf.flags.DEFINE_integer(name="filter_size", default=3, help="cnn filter size")
tf.flags.DEFINE_integer(name="filter_num", default=128, help="cnn filter num")

tf.flags.DEFINE_string(name="dataset_flag", default="end", help="split dataset sentence by end")
tf.flags.DEFINE_string(name="tag_char", default="O,B-S-ORG,I-S-ORG,E-S-ORG", help="used in dataset, split by ,")

tf.flags.DEFINE_string(name="saved_model", default="model", help="saved train model path, default ./model")

FLAG = tf.flags.FLAGS


def main(_):
    model = RnnCnnCrf(FLAG)
    model.train(FLAG)
    model.test(FLAG)


if __name__ == "__main__":
    tf.app.run(main)
