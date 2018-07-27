import json
import codecs
import os


class PrepareTagData(object):
    def __init__(self):
        prj = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
        self.trainPath = os.path.join(prj, "data/trainset")
        self.chineseVocab = os.path.join(prj, "data/vocab_dict.txt")
        self.dataSet = os.path.join(prj, "data/dataset.txt")
        self.vocabDict = self._load_chinese_vocab()
        self.tagId = self._tag_id()

    def _load_chinese_vocab(self):
        with codecs.open(self.chineseVocab, "r", "utf8") as f:
            return json.loads(f.read())

    @staticmethod
    def _tag_id():
        """
        O other
        B begin
        I
        E end
        :return:
        """
        return {
            "O": 0,
            "B-S-ORG": 1,
            "I-S-ORG": 2,
            "E-S-ORG": 3,
        }

    def train_data(self):
        sentence_lst = self._load_train_tag()
        return self.__split_train_data(sentence_lst)

    @staticmethod
    def __split_train_data(sentence_lst):
        inputs_x = []
        inputs_y = []
        for sentence in sentence_lst:
            inputs_x.append([item[0] for item in sentence])
            inputs_y.append([item[1] for item in sentence])
        return inputs_x, inputs_y

    def _load_train_tag(self):
        with codecs.open(self.trainPath, "r", "utf8") as f:
            sentence_lst = []
            char_lst = []
            for line in f.readlines():
                line = line.replace("\n", "").replace("B-ORG", "O").replace("I-ORG", "")
                if line != "end":
                    line = line.split(" ")
                    vocab_id = self.vocabDict.get(line[0], -1)
                    if vocab_id == -1:
                        continue
                    tag_id = self.tagId.get(line[1], -1)
                    if tag_id == -1:
                        continue
                    char_lst.append([vocab_id, tag_id])
                else:
                    sentence_lst.append(char_lst)
                    char_lst = []

        # with codecs.open(self.dataSet, "w", "utf8") as f:
        #     f.write(json.dumps(sentence_lst, indent=2, ensure_ascii=False))
        return sentence_lst
