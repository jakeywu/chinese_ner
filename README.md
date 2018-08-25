# 基于字符的LSTM+CRF中文实体抽取 

### 环境　推荐[Anaconda mini管理](https://www.jianshu.com/p/169403f7e40c)

python3.6.5 

tensorflow==1.8.0

### 项目文件.
1. data_utils.py 为数据预处理文件. 重写__next__方法直接遍历数据对象得到批次数据
2. rnn_cnn_crf.py 为lstm_crf模型文件, 接下来会添加`cnn`特征提取, 以解决句子局部关系
3. train_model.py 模型加载文件.
4. predict.py 模型预测文件, 其中模型采用saved_model模块存储, 方便采用A/B测试时 server端稳定

### 效果
 <<Train>> EPOCH: [2] Iter: [11244] STEP: [719573] LOSS: [0.982]       [acc: 0.903 recall: 0.845 f1: 0.873]
11249 <<Train>> EPOCH: [2] Iter: [11245] STEP: [719637] LOSS: [0.853]       [acc: 0.940 recall: 0.916 f1: 0.928]
11250 <<Train>> EPOCH: [2] Iter: [11246] STEP: [719701] LOSS: [0.823]       [acc: 0.922 recall: 0.905 f1: 0.913]
11251 <<Train>> EPOCH: [2] Iter: [11247] STEP: [719765] LOSS: [1.064]       [acc: 0.897 recall: 0.870 f1: 0.883]
11252 <<Train>> EPOCH: [2] Iter: [11248] STEP: [719829] LOSS: [0.485]       [acc: 0.976 recall: 0.892 f1: 0.933]
11253 <<Train>> EPOCH: [2] Iter: [11249] STEP: [719893] LOSS: [0.864]       [acc: 0.859 recall: 0.932 f1: 0.894]
11254 <<Train>> EPOCH: [2] Iter: [11250] STEP: [719957] LOSS: [0.770]       [acc: 0.907 recall: 0.924 f1: 0.915]
11255 <<Train>> EPOCH: [2] Iter: [11251] STEP: [720021] LOSS: [0.755]       [acc: 0.895 recall: 0.922 f1: 0.908]
11256 <<Train>> EPOCH: [2] Iter: [11252] STEP: [720085] LOSS: [0.823]       [acc: 0.874 recall: 0.928 f1: 0.900]
11257 <<Train>> EPOCH: [2] Iter: [11253] STEP: [720149] LOSS: [1.137]       [acc: 0.851 recall: 0.871 f1: 0.860]
11258 <<Train>> EPOCH: [2] Iter: [11254] STEP: [720213] LOSS: [0.873]       [acc: 0.845 recall: 0.934 f1: 0.888]
11259 <<Train>> EPOCH: [2] Iter: [11255] STEP: [720277] LOSS: [0.761]       [acc: 0.939 recall: 0.837 f1: 0.885]
11260 <<Train>> EPOCH: [2] Iter: [11256] STEP: [720341] LOSS: [1.010]       [acc: 0.964 recall: 0.810 f1: 0.880]
11261 <<Train>> EPOCH: [2] Iter: [11257] STEP: [720405] LOSS: [0.902]       [acc: 0.944 recall: 0.842 f1: 0.890]
11262 <<Train>> EPOCH: [2] Iter: [11258] STEP: [720469] LOSS: [1.025]       [acc: 0.895 recall: 0.913 f1: 0.904]
11263 <<Train>> EPOCH: [2] Iter: [11259] STEP: [720533] LOSS: [1.411]       [acc: 0.918 recall: 0.796 f1: 0.852]
11264 <<Train>> EPOCH: [2] Iter: [11260] STEP: [720554] LOSS: [1.001]       [acc: 0.909 recall: 0.800 f1: 0.851]

### 相关文档
[Tensor saved_model模块](https://blog.csdn.net/thriving_fcl/article/details/75213361)

[RNN CRF论文](https://www.aclweb.org/anthology/N16-1030)

[RNN CRF博客](https://www.cnblogs.com/Determined22/p/7238342.html)
