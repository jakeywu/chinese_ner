import tensorflow as tf
from data_utils import PrepareTagData
from train_model import FLAG

evalset = PrepareTagData(FLAG, "eval")

with tf.Graph().as_default():
    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAG.saved_model)
        signature = meta_graph_def.signature_def
        inputs = signature["ner_name"].inputs["inputs"].name
        keep_prob = signature["ner_name"].inputs["keep_prob"].name
        decode_tags = signature["ner_name"].inputs["decode_tags"].name
        best_score = signature["ner_name"].inputs["best_score"].name

        eval_x = sess.graph.get_tensor_by_name(inputs)
        eval_keep_prob = sess.graph.get_tensor_by_name(keep_prob)

        while True:
            try:
                input_x, _ = next(evalset)
                tags, score = sess.run([decode_tags, best_score], feed_dict={eval_x: input_x, keep_prob: 1})
            except Exception as e:
                import pdb
                pdb.set_trace()
                exit()
