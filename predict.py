import tensorflow as tf
from data_utils import PrepareTagData
from train_model import FLAG

evalset = PrepareTagData(FLAG, "eval")

with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAG.saved_model)
    signature = meta_graph_def.signature_def
    inputs = signature["inputs"].inputs["inputs"].name
    keep_prob = signature["inputs"].inputs["keep_prob"].name
    decode_tags = signature["outputs"].inputs["decode_tags"].name
    best_score = signature["outputs"].inputs["best_score"].name

    eval_x = sess.graph.get_tensor_by_name(inputs)
    eval_keep_prob = sess.graph.get_tensor_by_name(keep_prob)

    x = sess.graph.get_tensor_by_name('input_x:0')
    y = sess.graph.get_tensor_by_name('predict_y:0')

    sess.run([decode_tags, best_score], feed_dict={eval_x: [], keep_prob: 1})



