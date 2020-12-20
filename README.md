# sentiment-analysis 文本分类 / 情感分析

所有模型输出的都是(batch_size,)维度的张量，损失函数均采用`nn.BCEWithLogitsLoss()`。

## **词向量平均(word averaging)二分类模型**

[from_scratch.ipynb](https://nbviewer.jupyter.org/github/adengz/sentiment-analysis/blob/main/from_scratch.ipynb) [4] ~ [10]

输入序列经过embedding层后，沿seq_len轴用mean压缩所有词向量，再由全连接层映射成实数轴上的logit。

### 模型/训练参数

* EMBED_DIM = 256
* DROPOUT = 0.25
* BATCH_SIZE = 256
* OPTIM_CLS = Adam
* LR = 2e-4
* EPOCHS = 20

### 最佳模型

DEV最佳准确率：81.42%

TEST准确率：81.38%

### 词向量的L2-norm

```plain
worst         4.024570
suffers       3.295249
mess          3.271896
lacking       3.261830
remarkable    3.245809
lacks         3.195105
flat          3.192154
powerful      3.182127
devoid        3.166308
hilarious     3.112350
captures      3.098900
touching      3.046976
waste         2.944949
stupid        2.901314
terrific      2.898190
```
norm最大的15个词，感情色彩极其强烈，对模型的输出会有较大影响。

```plain
<pad>           0.000000
<unk>           0.017731
jesus           0.067001
non-bondish     0.067320
fustily         0.078295
reviews         0.079019
grounding       0.081172
malapropisms    0.081883
margarita       0.082631
fuelled         0.083875
freud           0.084031
bearing         0.086743
presentation    0.087168
by-the-book     0.088188
wattage         0.088215
```
norm最小的15个词，包括一些人名，很难阅读出任何感情色彩，对模型的输出影响很小。

## **Attention 加权平均(Attention Weighted word averaging)**

[from_scratch.ipynb](https://nbviewer.jupyter.org/github/adengz/sentiment-analysis/blob/main/from_scratch.ipynb) [11] ~ [21]

在前一任务模型的基础上，加上一个(embed_dim,)的向量u，用来计算每个输入词向量的权重。权重的值正比于u和词向量的余弦相似度。沿seq_len轴用softmax归一化后，将带权重的词向量沿seq_len轴求加权平均，再经过全连接层得到logit。

### 模型/训练参数

* EMBED_DIM = 256
* DROPOUT = 0.25
* BATCH_SIZE = 256
* OPTIM_CLS = Adam
* LR = 1e-4
* EPOCHS = 20

### 最佳模型

DEV最佳准确率：80.05%

TEST准确率：79.24%

### 词向量与向量u的的余弦相似度

```plain
never              0.996969
disappoint         0.992830
ignored            0.992009
patriotic          0.991844
bowl               0.991654
talkiness          0.989554
afraid             0.989487
mcadams            0.989437
rah-rah            0.988890
superman           0.988618
broadcast          0.988453
unintentionally    0.988083
moan               0.987819
miss               0.987211
wrong              0.986977
```
许多感情色彩强烈的词都与u有很高的相似度。

```plain
quick         -0.999458
adults        -0.999302
have          -0.999283
there         -0.999252
jones         -0.999176
rather        -0.999142
time          -0.999133
back          -0.999114
narratively   -0.999103
making        -0.999089
understand    -0.999076
segment       -0.999049
veiling       -0.999038
begins        -0.999025
bit           -0.999008
```
相反，相似度低的词较难从字面分析出感情色彩。

### Attention的变化

|word|mean|std|
|:----:|:----:|:----:|
|bland|0.346257|0.265820|
|stupid|0.385331|0.265657|
|awful|0.413376|0.255618|
|painful|0.372556|0.252404|
|tedious|0.325079|0.248903|
|flat|0.327273|0.245703|
|creepy|0.332419|0.236097|
|worse|0.296377|0.235868|
|waste|0.378752|0.235385|
|unfunny|0.335676|0.233512|
|boring|0.337892|0.232801|
|bad|0.347086|0.231796|
|tired|0.367621|0.230907|
|hackneyed|0.307232|0.230533|
|mess|0.387299|0.228345|
|lacking|0.347283|0.227518|
|worst|0.369018|0.222505|
|unsettling|0.381576|0.221678|
|dumb|0.337494|0.220777|
|tragedy|0.297227|0.218020|
|impossible|0.302223|0.216868|
|lacks|0.382707|0.216808|
|dull|0.348571|0.216652|
|shallow|0.306856|0.215912|
|pretentious|0.319199|0.214811|
|death|0.271942|0.212622|
|slow|0.342638|0.211525|
|cold|0.331609|0.210278|
|fails|0.320970|0.209626|
|cheap|0.297314|0.209507|

Attention标准差很大的词，也大多含有极端的感情色彩。要注意的是，attention的计算中，句子长短也会有所贡献，一般句子越短softmax后的值越大。这里mean和std很可能比较大的相关度，有待进一步分析证实。

## Self Attention机制

[from_scratch.ipynb](https://nbviewer.jupyter.org/github/adengz/sentiment-analysis/blob/main/from_scratch.ipynb) [22] ~ [29]

在前一模型的基础上，改变权重计算的方式，不再依赖于额外的可训练的u向量，而是采用当前词向量与句子中其它词向量的点积。沿序列长度轴用softmax归一化后，将带权重的词向量再沿序列长度轴求加权平均。若打开残差连接，则须在加权平均后的词向量上再加上（无权重的）词向量平均。最终经过全连接层得到logit。

### 模型/训练参数

* EMBED_DIM = 256
* DROPOUT = 0.25
* BATCH_SIZE = 256
* OPTIM_CLS = Adam
* LR = 5e-5
* EPOCHS = 20

### 最佳模型

|残差连接|DEV最佳准确率|TEST准确率|
|:----:|:----:|:----:|
|无|80.50%|80.94%|
|有|80.85%|81.88%|

### 分析

自注意力模型的表现意外地略好于使用了额外u向量的注意力模型。两种求权重的方式有相似之处，本质上都存在点积的计算，但余弦相似度会消除向量范数的影响，能适当平衡较大范数向量对输出结果的影响。点积自注意力模型再添加残差连接后，模型效果也有进一步提升。

## 设计Attention函数

[from_scratch.ipynb](https://nbviewer.jupyter.org/github/adengz/sentiment-analysis/blob/main/from_scratch.ipynb) [30] ~ [45]

这里的模型在词向量与最终的全连接层之间采用了带残差连接的Multi-Head Attention。生成的词向量可选择性地加上位置编码。

### 模型/训练参数

* EMBED_DIM = 256
* DROPOUT = 0.25
* BATCH_SIZE = 256
* OPTIM_CLS = Adam
* LR = 5e-6
* EPOCHS = 20

### 最佳模型

|“注意力头”数|位置编码|DEV最佳准确率|TEST准确率|
|:----:|:----:|:----:|:----:|
|1|无|81.77%|81.93%|
|1|有|81.31%|81.99%|
|4|无|80.96%|81.44%|
|4|有|81.88%|81.60%|

### 分析

Transformer中的Multi-Head Attention具有很强的学习能力，为此特意选了较小的学习率防止过快过拟合。整体效果相比之前的模型都略好，但在小数据集面前也很难有大的突破。能引发进一步量变的，是利用这类学习能力强的模型在海量数据上做预训练。

## Bert模型

[pre_trained.ipynb](https://nbviewer.jupyter.org/github/adengz/sentiment-analysis/blob/main/pre_trained.ipynb)

用预训练BERT模型及其下游任务的处理方式来做文本分类。

### 训练参数

* BATCH_SIZE = 64
* OPTIM_CLS = AdamW
* LR = 1e-6
* EPOCHS = 10

### 最佳模型

DEV最佳准确率：91.40%

TEST准确率：91.60%

### 分析

用预训练模型来做文本分类基本就是降维打击了。前面从头训练的模型最后取得的效果可能不比传统机器学习模型高出很多，但预训练模型把所有准确率都拉高了10%左右。训练以及预测的成本可能会是限制预训练模型大规模应用的瓶颈，在对模型运行效率要求高的场景下（in-time inference），简单轻量的模型会依旧更受欢迎。

