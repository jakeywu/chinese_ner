# 基于字符的LSTM+CRF中文实体抽取 

### 环境　推荐[Anaconda mini管理](https://www.jianshu.com/p/169403f7e40c)

python3.6.5 

tensorflow==1.8.0

### 项目文件.
1. data_utils.py 为数据预处理文件. 重写__next__方法直接遍历数据对象得到批次数据
2. rnn_cnn_crf.py 为lstm_crf模型文件, 接下来会添加`cnn`特征提取, 已解决句子局部关系
3. train_model.py 模型加载文件.
4. predict.py 模型预测文件, 其中模型采用saved_model模块存储, 方便采用A/B测试时 server端稳定

### 相关文档
[Tensor saved_model模块](https://blog.csdn.net/thriving_fcl/article/details/75213361)

[RNN CRF论文](https://www.aclweb.org/anthology/N16-1030)

[RNN CRF博客](https://www.cnblogs.com/Determined22/p/7238342.html)