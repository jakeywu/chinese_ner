import tensorflow as tf
from model.data_utils import PrepareTagData
from model.rnn_cnn_crf import RnnCnnCrf

tf.flags.DEFINE_integer(name="num_tag", default=4, help="number tags")
tf.flags.DEFINE_integer(name="epoch", default=10, help="maximum epochs")
tf.flags.DEFINE_integer(name="vocab_size", default=5000, help="vocab num")
tf.flags.DEFINE_integer(name="num_hidden", default=128, help="lstm num hidden")
tf.flags.DEFINE_integer(name="embedding_size", default=128, help="embedding size")
tf.flags.DEFINE_integer(name="filter_size", default=3, help="cnn filter size")
tf.flags.DEFINE_integer(name="filter_num", default=128, help="cnn filter num")
FLAG = tf.flags.FLAGS


if __name__ == "__main__":
    train_x, train_y = PrepareTagData().train_data()
    model = RnnCnnCrf(FLAG)
    import pdb
    pdb.set_trace()
    tf.app.run()
