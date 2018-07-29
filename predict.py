import os
import json
import codecs
import tensorflow as tf


class PredictNer(object):
    def __init__(self, saved_model, source_data):
        graph = tf.Graph()
        self.seved_model = saved_model
        self.chinese_vocab = self.__load_chinese_vocab()
        self.source_data = source_data
        self.__pre_handle_sentence()
        self.sess = tf.Session(graph=graph)
        self.__get_tensor_name()

    def __get_tensor_name(self):
        meta_graph_def = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.seved_model)
        signature = meta_graph_def.signature_def
        self.input_x = self.sess.graph.get_tensor_by_name(signature["ner_name"].inputs["inputs_x"].name)
        self.keep_prob = self.sess.graph.get_tensor_by_name(signature["ner_name"].inputs["keep_prob"].name)
        self.decode_tags = self.sess.graph.get_tensor_by_name(signature["ner_name"].outputs["decode_tags"].name)

    @staticmethod
    def __load_chinese_vocab():
        cv = dict()
        with codecs.open(os.path.join(os.path.dirname(__file__), "data/chinese_vocab.txt"), "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                cv[line.strip()] = i
        return cv

    def __pre_handle_sentence(self):
        input_x = []
        for _text in self.source_data:
            sentence_lst = []
            for s in _text:
                _id = self.chinese_vocab.get(s, -1)
                if _id == -1:
                    raise Exception("count of chinese vocab not enough")
                sentence_lst.append(_id)
            input_x.append(sentence_lst)
        max_len = max([len(x) for x in input_x])
        self.content = tf.keras.preprocessing.sequence.pad_sequences(
            input_x, maxlen=max_len, padding="post", truncating="post", dtype="int32", value=0)

    def batch_predict_ner(self):
        decode_tags = self.sess.run(self.decode_tags, feed_dict={self.input_x: self.content, self.keep_prob: 1.})
        result_ner = []
        for i in range(len(decode_tags)):
            sentence_ner = []
            for j in range(len(decode_tags[i])):
                if decode_tags[i].tolist()[j] == 0:
                    continue
                sentence_ner.append(self.source_data[i].tolist()[j])
            result_ner.append(sentence_ner)
        return result_ner


if __name__ == "__main__":
    content = [
        "美国Groupon公司曾经是团购商业模式的发明者和行业龙头，曾经引发中国市场的百团大战、千团大战。不过这家公司如今陷入了"
        "寻求变卖的困境中。据外媒最新消息，Groupon迎来又一个坏消息，法庭裁决该公司侵犯了IBM的电子商务专利，必须赔偿8250万"
        "美元。",
        "美国Groupon公司曾经是团购商业模式的发明者和行业龙头，曾经引发中国市场的百团大战、千团大战。不过这家公司如今陷入了"
        "寻求变卖的困境中。据外媒最新消息，Groupon迎来又一个坏消息，法庭裁决该公司侵犯了IBM的电子商务专利，必须赔偿8250万"
        "美元。"
    ]
    ner = PredictNer("model", content)
    pred = ner.batch_predict_ner()
    print(json.dumps(pred, indent=2, ensure_ascii=False))
