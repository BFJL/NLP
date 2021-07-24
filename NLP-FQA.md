# 数据预处理

## **怎么获取训练样本**

获取高质量的标注数据需要花费昂贵的人力、物力，因此引出很多其他的学习方式，比如半监督、无监督、远程监督、迁移学习等等。

**远程监督（Distant Supervision）算法**，该算法的核心思想是将文本与大规模知识图谱进行实体对齐，利用知识图谱已有的实体间关系对文本进行标注。

远程监督基于的基本假设是：如果从知识图谱中可获取三元组R（E1，E2）（注：R代表关系，E1、E2代表两个实体），且E1和E2共现与句子S中，则S表达了E1和E2间的关系R，标注为训练正例。

该算法很好地解决了数据标注的规模问题，但它基于的基本假设过强，会引入大量噪音数据。训练集会产生大量的 wrong labels，比如两个实体有多种关系或者根本在这句话中没有任何关系，这样的训练数据会对关系抽取器产生影响。

## 数据集不平衡的处理方法

数据不均衡（某一标签数量太多，其余标签数量太少）的问题，在机器学习中被称为**“长尾问题”**。

**（1）扩充数据集**

　　　首先想到能否获得更多数据，尤其是小类（该类样本数据极少）的数据。

**（2）对数据集进行重采样**

　　**a）过采样（over-sampling）：**对小类的数据样本进行过采样来增加小类的数据样本个数。过采样容易发生少样本过拟合，无法学习更鲁棒、易泛化的特征，在不平衡数据上表现较差。为了解决这一问题，可以在每次生成新数据点时加入轻微的随机扰动，经验表明这种做法非常有效，但是这一方式会加重过拟合。

​		**b）欠采样（under-sampling）：**对大类的数据样本进行欠采样来减少大类的数据样本个数，欠采样会造成多样本严重信息损失，导致发生欠拟合。

**（3）人造数据**

　　**a）属性值随机采样**：在该类下所有样本的每个属性特征的取值空间中随机选取一个组成新的样本，即属性值随机采样。此方法多用于小类中的样本，但是该方法可能会产生现实中不存在的样本

​		**b）SMOTE(Synthetic Minority Over-sampling Technique)**

　　　SMOTE是一种过采样算法，它构造新的小类样本而不是产生小类中已有的样本的副本

**（4）改变分类算法**

　　**a）从损失函数的层面考虑**

**重加权** （re-weighting）：使用代价函数时，可以增加小类样本的权值，降低大类样本的权值（这种方法其实是产生了新的数据分布，即产生了新的数据集），从而使得分类器将重点集中在小类样本身上。刚开始，可以设置每个类别的权值与样本个数比例的倒数

　　**b）转化为异常点检测问题**

可以把小类样本作为异常点(outliers)，把问题转化为异常点检测问题(anomaly detection)。此时分类器需要学习到大类的决策分界面，即分类器是一个单个类分类器（One Class Classifier）

​		**c）focal loss**

**（5）迁移学习** （transfer learning）：对多类和少类样本分别建模，将学到的多类样本信息/表示/知识迁移给少类别使用。

**（6）解耦特征和分类器** （decoupling representation & classifier）：研究发现，将特征学习和分类器学习解耦、将不平衡学习分为两个阶段，并在特征学习阶段正常采样、在分类器学习阶段平衡采样，可以带来更好的长尾学习效果。这是目前最优的长尾分类算法。

## 数据增强的实现方法

数据量很小怎么办？

**简单数据增强(Easy Data Augmentation，EDA)**

**同义词替换(Synonym Replacement, SR)**：从句子中随机选取n个不属于停用词集的单词，并随机选择其同义词替换它们；

**随机插入(Random Insertion, RI)**：随机的找出句中某个不属于停用词集的词，并求出其随机的同义词，将该同义词插入句子的一个随机位置。重复n次；

**随机交换(Random Swap, RS)**：随机的选择句中两个单词并交换它们的位置。重复n次；

**随机删除(Random Deletion, RD)**：随机的移除句中的每个单词。

**回译**：反向翻译是NLP在机器翻译中经常使用的一个数据增强的方法。。其本质就是快速产生一些不那么准确的翻译结果达到增加数据的目的。回译的方法不仅有类似同义词替换的能力，它还具有在保持原意的前提下增加或移除单词并重新组织句子的能力。

**文档裁剪**：新闻文章通常很长，在查看数据时，对于分类来说并不需要整篇文章。 文章的主要想法通常会重复出现。将文章裁剪为几个子文章来实现数据增强，这样将获得更多的数据。

**生成对抗网络**：GAN是深度学习领域中最令人兴奋的最新进展之一，它们通常用来生成新的图像，但它的一些方法或许也可以适用于文本数据。

**数据增强的作用**

（1）增加训练的数据量，提高模型的泛化能力。

（2）增加噪声数据，提升模型的鲁棒性。

## 重复文本生成怎么解决

定义一个损失函数，**明确地**对“重复”做出惩罚。

模式崩溃。

## 怎么处理没有标签的数据？

远程监督

迁移学习：one-shot zero-shot 

label embedding：将一个label下的样本归纳成一个向量，新的预测样本，encode之后，与这个向量去计算一个score，最后得到该样本的predict label。

# 算法模型

## 传统深度学习

RNN能够更好的处理序列的信，**它能挖掘数据中的时序信息以及语义信息**。![img](NLP-FQA_images/v2-b0175ebd3419f9a11a3d0d8b00e28675_b.jpg)

![img](NLP-FQA_images/LSTM3-SimpleRNN.png)RNN存在什么的问题？RNN一个最大的缺陷就是**梯度消失与梯度爆炸问题**。无论是梯度消失还是梯度爆炸，都是**源于网络结构太深**，造成网络权重不稳定，从本质上来讲是**因为梯度反向传播中的连乘效应。**

**长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN**，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。LSTM主要用来解决RNN（循环神经网络）中存在的长期依赖问题。

![A LSTM neural network.](NLP-FQA_images/LSTM3-chain.png)

1. 输入：h_t-1，x_t，c_t-1
2. 首先h_t-1、x_t 经过遗忘门更新记忆单元c_t-1，决定哪些信息被遗忘
3. 接着h_t-1、x_t 经过输入门决定哪些信息以多大比例被增加到记忆单元中，此时记忆单元更新为c_t
4. 最后h_t-1、x_t 经过输出门，根据当前记忆c_t决定下一时刻的输出h_t
5. 输出：h_t（输出），h_t（隐藏状态），c_t

具有三个门：

1. 遗忘门
2. 输入门
3. 输出门

激活函数函数的作用：LSTM中的三个门是用的sigmoid作为激活函数，生成候选记忆时候用的才是tanh，门j的激活函数如果用relu的话会有个问题，就是relu是没有饱和区域的，那么就没法起到门的作用。这两个激活函数都是**饱和**的。

候选记忆用tanh是因为tanh的输出在-1~1，是0中心的，并且在0附近的梯度大，模型收敛快

优缺点：

方便序列建模，具备长时记忆的能力。

RNN本身的序列依赖结构对于大规模并行计算来说相当的不友好。更长的序列则依然会显得很棘手。

**GRU(Gated Recurrent Unit)是一种LSTM的变形版本**，它将遗忘和输入门组合成一个“更新门”。它还合并了单元状态和隐藏状态。![img](NLP-FQA_images/1470684-20180927210004968-2129013009.png)

28、循环神经网络RNN 怎么解决长期依赖问题？LSTM 的结构是怎样的？

29、怎么理解“长短时记忆单元”？RNN 中的隐状态ht 与LSTM 中的记忆状态Ct 有什么区别？

## Embedding

one-hot**过于稀疏时，过度占用资源**

Embedding 字面理解是 “嵌入”，实质是一种映射，从语义空间到向量空间的映射，同时尽可能在向量空间保持原样本在语义空间的关系，如语义接近的两个词汇在向量空间中的位置也比较接近。

embedding 的原理和在项目中的具体使用步骤

one-hot与bert embedding 的区别

3个embedding介绍

positional embedding怎么生成的（cos，sin生成的，相当于一个look up tabel）

positional embedding公式

embedding怎么实现？（fasttext or word2vec，不能用BERT，数据量大不合适）补充：你说的是词向量，句子向量怎么做呢（+average pooling最常用，还有别的方法没列举）

embedding中参数的传递和计算

embedding如何学到语音信息，为什么不用拼音来学。

15、embedding 的作用是什么？

16、embedding 物理意义?

## word2vec

Word2vec，就是词嵌入（ word embedding) 的一种。

- 如果是用一个词语作为输入，来预测它周围的上下文，那这个模型叫做『Skip-gram 模型』

  Skip-gram是用中心词来预测周围的词。在skip-gram中，会利用周围的词的预测结果情况，使用GradientDecent来不断的调整中心词的词向量，最终所有的文本遍历完毕之后，也就得到了文本所有词的词向量。

  可以看出，Skip-gram进行预测的次数是要多于CBOW的：因为每个词在作为中心词时，都要使用周围词进行预测一次。这样相当于比CBOW的方法多进行了K次（假设K为窗口大小），因此时间的复杂度为O(KV)，训练时间要比CBOW要长。

  但是在Skip-gram当中，每个词都要收到周围的词的影响，每个词在作为中心词的时候，都要进行K次的预测、调整。因此， 当数据量较少，或者词为生僻词出现次数较少时， 这种多次的调整会使得词向量相对的更加准确。因为尽管CBOW从另外一个角度来说，某个词也是会受到多次周围词的影响（多次将其包含在内的窗口移动），进行词向量的跳帧，但是他的调整是跟周围的词一起调整的，grad的值会平均分到该词上， 相当于该生僻词没有收到专门的训练，它只是沾了周围词的光而已。

  鉴于Skip-gram学习的词向量更细致，但语料库中有大量低频词时，使用skip-gram学习比较合适

- 而如果是拿一个词语的上下文作为输入，来预测这个词语本身，则是 『CBOW (Continuous Bag-of-Words) 模型』

  在CBOW方法中，是用周围词预测中心词，从而利用中心词的预测结果情况，使用GradientDesent方法，不断的去调整周围词的向量。当训练完成之后，每个词都会作为中心词，把周围词的词向量进行了调整，这样也就获得了整个文本里面所有词的词向量。

  要注意的是， CBOW的对周围词的调整是统一的：求出的gradient的值会同样的作用到每个周围词的词向量当中去。

  可以看到，CBOW预测行为的次数跟整个文本的词数几乎是相等的（每次预测行为才会进行一次backpropgation, 而往往这也是最耗时的部分），复杂度大概是O(V);

从更通俗的角度来说：

在Skip-gram里面，每个词在作为中心词的时候，实际上是 1个学生 VS K个老师，K个老师（周围词）都会对学生（中心词）进行“专业”的训练，这样学生（中心词）的“能力”（向量结果）相对就会扎实（准确）一些，但是这样肯定会使用更长的时间；

CBOW是 1个老师 VS K个学生，K个学生（周围词）都会从老师（中心词）那里学习知识，但是老师（中心词）是一视同仁的，教给大家的一样的知识。至于你学到了多少，还要看下一轮（假如还在窗口内），或者以后的某一轮，你还有机会加入老师的课堂当中（再次出现作为周围词），跟着大家一起学习，然后进步一点。因此相对Skip-gram，你的业务能力肯定没有人家强，但是对于整个训练营（训练过程）来说，这样肯定效率高，速度更快。



为了提高速度，Word2vec 经常采用 2 种加速方式：

1. Negative Sample（负采样）？
2. Hierarchical Softmax ？

优点：

1. 由于 Word2vec 会考虑上下文，跟之前的 Embedding 方法相比，效果要更好（但不如 18 年之后的方法）
2. 比之前的 Embedding方 法维度更少，所以速度更快
3. 通用性很强，可以用在各种 NLP 任务中

缺点：

1. 由于词和向量是一对一的关系，所以多义词的问题无法解决。
2. Word2vec 是一种静态的方式，虽然通用性强，但是无法针对特定任务做动态优化

CBOW比Skip-gram训练速度快，训练过程更加稳定，原因是CBOW使用上下文average的方式进行训练，每个训练step会见到更多样本。而在生僻字（出现频率低的字）处理上，skip-gram比CBOW效果更好，原因是skip-gram不会刻意回避生僻字。

22、word2vec 算法推导

23、word2vec 的优化

24、glove 和word2vec 对比有什么区别？

### Skip-gram的实际实现

[训练中文词向量Word2vec -Skip-gram-负采样](https://aistudio.baidu.com/aistudio/projectdetail/2158141)

### word2vec-FQA

word2vec的训练相比bert的情况

word2Vec模型，对于其中的窗口选择策略？窗口选择5，10，15都有什么区别吗？是不是越大越好呢？项目中的一些调参的方法是怎样做得呢？

对golve word2vec 的理解

word2vec 结构中向量提取的位置

word2vec的模型、训练目标

word2vec跟其他预训练模型的区别

word2vec的优化，包括层级softmax的复杂度

word2vec与bert 的区别

word2vec和glove的原理和使用场景，差别

word2vec如何做负采样？是在全局采样？还是在batch采样？如何实现多batch采样？怎么确保采样不会采到正样本？

word2vec负采样时为什么要对频率做3/4次方？

word2vec经过霍夫曼或者负采样之后，模型与原模型相比，是等价的还是相似的？

## Attention

多头注意力机制的原理是什么？

具体讲一下self attention。

讲一下attention。

self attention， attention， 双向lstm的区别。

attention和self-attention的公式

attention的公式， 为什么除根号dk

Self-attention的Query，Key，Value分别是什么

self-attention的乘法计算和加法计算有什么区别？什么时候乘比较好，什么时候加？为什么要除以一个根号？

bert中qkv三个向量是否共享，意义

multi-head attention的作用与attention的区别

剪枝的原理，为什么多头attention剪枝有用

attention mask的作用

q、k、v是啥意思（解释了一下）qk可以用别的方法代替吗（可以，只要是计算距离的方法就行，比如说余弦距离）

self-attention和attention的区别（说了一堆好像没答道点上，最核心应该是attention只计算q和v）

32、Encoder-Decoder attention 与self-attention

33、Multi-Head attention、多头的理解

34、Position Embedding 位置编码

## Transformer

**介绍一下transformer？**

Transformer 本身还是一个典型的 encoder-decoder 模型，如果从模型层面来看，Transformer 实际上就像一个 seq2seq with attention 的模型。

参数：Encoder 6层，Decoder 6层，隐层512维，Forward 2048，多头8个，K V矩阵维度64（512/8），drop 0.1， Label Smoothing

**transformer为何使用多头注意力机制？**

多头的注意力**有助于网络捕捉到更丰富的特征/信息。使用多头注意力，也就是综合利用各方面的信息/特征**。还可以正则化手段来保证每个头关注不同的子空间（[Multi-Head Attention with Disagreement Regularization](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1810.10183)）

**Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？**

两个向量的点乘表示两个向量的相似度， 如果在同一个向量空间里进行点乘，理所应当的是自身和自身的相似度最大，那会影响其他向量对自己的作用，使用Q/K/V 不相同，保证在不同空间进行投影，增强表达能力，提高泛化能力。

**Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？**

为了计算更快。矩阵加法在加法这一块的计算量确实简单，但是作为一个整体计算attention的时候相当于一个隐层，整体计算量和点积相似。在效果上来说，从实验分析，两者的效果和dk相关，dk越大，加法的效果越显著。

**transformer 一个block中最耗时的部分？**

block里面包括 self-attention, layer-norm, feed-forward, self attention里面，要将三个enc映射到不同的子空间，然后再点乘，还要normlizar，所以self attention最耗时。

**为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解？**

- 维度越高，整个向量的方差越大，会出现很大的值和很小的值。
- 计算softmax的梯度时候，由于softmax的梯度导数，会导致梯度变小。
- [公式推导讲解](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_37430422/article/details/105042303)。

**计算attention score的时候如何对padding做mask操作？**

mask就是让这部分值取无穷小，让他再softmax之后基本也为0，不去影响非attention socore的分布

**介绍一下Transformer的位置编码？有什么意义和优缺点？**

当抛弃循环神经网络结构，完全采用Attention取而代之，这些词序信息就会丢失，模型就没有办法知道每个词在句子中的相对和绝对的位置信息。因此，有必要把词序信号加到词向量上帮助模型学习这些信息，位置编码（Positional Encoding）就是用来解决这种问题的方法。

位置编码（Positional Encoding）是一种用词的位置信息对序列中的每个词进行二次表示的方法。

```python
import numpy as np
import matplotlib.pyplot as plt

def get_angles(pos, i, d_model):
    """计算pos/100002i/dmodel"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    print('-----pos/100002i/dmodel--------')
    print(angle_rates)
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], 
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    print('---angle_rads---')
    print(angle_rads)
    # 对数组中的偶数下标2i应用sin
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 对数组中的奇数下标2i+1应用cos
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    print('-----sin--cos---')
    print(angle_rads)
    # 增加一个新得维度
    pos_encoding = angle_rads[np.newaxis, ...]
    print('----pos----')
    print(pos_encoding)
    return pos_encoding


tokens = 5
dimensions = 10

pos_encoding = positional_encoding(tokens, dimensions)
print (pos_encoding.shape)

plt.figure(figsize=(20,10))
plt.pcolormesh(pos_encoding[0], cmap='viridis')
plt.xlabel('Embedding Dimensions')
plt.xlim((0, dimensions))

plt.ylabel('Token Position')
plt.ylim((tokens,0))

plt.colorbar()
plt.show()
```

**为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm 在Transformer的位置是哪里，BN使用场景和layer_norm区别？**

对于使用场景来说，BN在MLP和CNN上使用的效果都比较好，在RNN这种动态文本模型上使用的比较差。

BN在MLP中的应用。 BN是对每个特征在batch_size上求的均值和方差，如果BN应用到NLP任务，相当于是在对默认了在同一个位置的单词对应的是同一种特征，

layer_norm针对的是文本的长度，整条序列的文本，所以比bn好。

**残差结构意义？**

防止梯度消失，帮助深层网络训练

**Transformer 相比于 RNN/LSTM，有什么优势？为什么？**

（1） RNN 系列的模型，并行计算能力很差， 但其实真实的耗时从业务上来讲，长query，transformer更快，短query，lstm更快。

（2）Transformer 的特征抽取能力比 RNN 系列的模型要好，transform可以动态建立输入序列的长程依赖关系，类似于一层全链接，

transformer中的参数共享

transformer结构图

## Bert

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder。

layer norm和 batch norm的区别，bert用的是哪个, bert中的norm是对那些输入进行的归一化,和各自的应用场景。已经各自在norm之后还有没有其他操作。训练时候和测试时候的区别。

BERT和其他序列模型区别（BERT不是序列模型，Auto-Encoder的模型，其他比如LSTM是Auto-Regression）

用过bert嘛，时间复杂度是多少？讲一下对self-attention的理解。

说一下BERT和transformer的positional embedding有啥区别

bert中feed forward中参数的传导

bert介绍（跟transformer差不多把，就是多了两个预训练任务

Bert模型的，历史、演变。（RNN、LSTM、ELMo、Attention、Transformer、GPT、BERT、Auto-Encoder、Auto-Regression、Positional Encoder）

Bert为什么效果这么好。

bert中的数据流动，Embedding是怎么做的

BERT模型怎么做的?大致的网络架构是怎么样的?BERT模型对防止梯度消失优化是什么?

word2Vec和BERT的词向量为什么BERT更好?

bert的局限性

还知道哪些bert模型

bert的参数规模，以及大部分参数所处的位置

bert与其他nlp模型的对比和使用情况

Bert，elmo，word2vec等词向量的区别

bert模型结构的具体介绍以及实现

bert自带的词向量库与glove的对比

bert三个embedding的使用方法

roberta和bert的区别

bert结构/ 与训练任务

对不同的bert蒸馏模型的理解
Bert预训练任务有哪些？为什么要这么做？你觉得哪些任务可以如何优化？

bert变种有了解么（介绍了一下roberta和xlnet针对mlm的问题的改进，讲了讲DisliBERT，

bert与roberta的不同

介绍一下BERT的三种的embedding的输入.
transformer的position embedding和BERT的position embedding的区别.

BERT和RNN之类的区别（Auto Encoder与Auto regression，吧啦吧啦 ）
3.BERT的缺点，什么情况无法处理，双重否定，数值可以处理吗（应该不太明感）
4.BERT的输入长度多少？（最大512） 超出怎么办（截断，数据预处理减少无关数据。实在太长直接上LSTM、Transforemer-XL）

LSTM和BERT模型有什么区别?

35、Bert 训练的一些参数？

36、bert 算是去年的工作了，有了解今年有哪些新的技术吗？简单讲讲。

37、BERT 原理。

38、BERT 适合哪些场景，不适合哪些场景。

39、BERT 训练过程中用到涉及到的哪些任务，这些任务如何去实现完成的。

40、如何使用bert？Bert 调参数的心得。



# 优化

## Dropout

**Dropout原理**：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，减少过拟合，因为它不会太依赖某些局部的特征。缺点就是代价函数不再被明确定义，每次迭代，都会随机移除一些节点，如果再三检查梯度下降的性能，实际上是很难进行复查的。

**Dropout在训练过程和测试过程中参数的设置？**

Dropout 在训练时采用，是为了减少神经元对部分上层神经元的依赖，类似将多个不同网络结构的模型集成起来，减少过拟合的风险，一般认为设置为0.5或者0.3。

而在测试时，应该用整个训练好的模型，因此不需要dropout。

**Dropout 如何平衡训练和测试时的差异呢？**

假设失活概率为 p ，就是这一层中的每个神经元都有p的概率失活，如下图的三层网络结构中，如果失活概率为0.5，则平均每一次训练有3个神经元失活，所以输出层每个神经元只有3个输入，而实际测试时是不会有dropout的，输出层每个神经元都有6个输入，这样在训练和测试时，输出层每个神经元的输入和的期望会有量级上的差异。

因此在训练时还要对第二层的输出数据除以（1-p）之后再传给输出层神经元，作为神经元失活的补偿，以使得在训练时和测试时每一层输入有大致相同的期望。

![img](NLP-FQA_images/v2-7133b2b708be2dab23c8917a1d60461d_720w.jpg)

## 拟合

偏差（Bias），是指一个模型在不同训练集上的平均性能和最优模型的差异，可以用来衡量一个模型的拟合能力．（低偏差靠近中心）

方差（Variance），是指一个模型在不同训练集上的差异，可以用来衡量一个模型是否容易过拟合．（低方差聚集）

方差一般会随着训练样本的增加而减少．当样本比较多时，方差比较少，这时可以选择能力强的模型来减少偏差．

期望错误

![image-20210525153908610](NLP-FQA_images/image-20210525153908610.png)

其中

![image-20210525153919761](NLP-FQA_images/image-20210525153919761.png)

**什么是过拟合和欠拟合？**

过拟合（Overfitting）：在训练集上错误率很低，但是在未知数据上错误率很高。高方差（**high variance**）

过拟合问题往往是由于训练数据少和噪声以及模型能力强等原因造成的。

欠拟合（Underfitting），即模型不能很好地拟合训练数据，在训练集上的错误率比较高。高偏差（**high bias**）

欠拟合一般是由于模型能力不足造成的。

**如何防止过拟合和欠拟合？**

防止过拟合：

- 模型层面降低复杂度

  - 模型本身的选择

  - 模型的参数个数
  - 模型的参数空间选择：参数的正则化（Regularization）


- 正则化

  - Dropout（随机失活）正则化

  - L1 正则化
  - L2 正则化 有时被称为“权重衰减”的原因
- 数据层面确保测试集训练集分布一致
- 降噪
- 模型集成
- 数据扩增
- 提前停止（early stopping）：缺点就是提早停止梯度下降，也就是停止了优化代价函数，所以代价函数的值可能不够小。
- 特征学习：在一定程度上减少模型复杂性、缩短训练时间、提高模型泛化能力、避免过拟合等。
- 经验风险最小化（Empirical Risk Minimization，ERM）准则。
- 结构风险最小化（Structure Risk Minimization，SRM）准则。


## 梯度

**梯度下降方法有哪些？怎么选择？**

**梯度消失和梯度爆炸原因分析及解决办法？**

激活函数，网络层次，正则化

**SGD的优点是什么？**

对于梯度的要求很低,计算比较快

# 激活函数

对激活函数的了解对比，为什么使用激活函数

sigmoid有哪些特性？激活函数了解多少？怎么选择

介绍一下sigmoid 和 relu，relu有什么缺点？

sigmoid当数据太大的时候不敏感怎么办（normalization）

常见的激活函数有哪些。各自有什么特点。分布应用于场景。leaky relu公式。

# 损失函数

loss设计 triplet loss和交叉熵loss各自的优缺点，怎么选择

交叉熵的损失函数是多少，大致是怎么算的。

对交叉熵的介绍和公式

交叉熵的公式

模型蒸馏的损失函数

不同损失函数之间的差异以及取舍问题

# 评估

**准确率** 最常用的评价指标为准确率（Accuracy）：

![image-20210525154620800](NLP-FQA_images/image-20210525154620800.png)

**错误率** 和准确率相对应的就是错误率（Error Rate）：

![image-20210525154653777](NLP-FQA_images/image-20210525154653777.png)

**精确率和召回率** 

准确率是所有类别整体性能的平均，如果希望对每个类都进行性能估计，就需要计算精确率（Precision）和召回率（Recall）．

1. 真正例（True Positive，TP）：一个样本的真实类别为𝑐并且模型正确地预测为类别𝑐．
2. 假负例（False Negative，FN）：一个样本的真实类别为𝑐，模型错误地预测为其他类．
3. 假正例（False Positive，FP）：一个样本的真实类别为其他类，模型错误地预测为类别𝑐
4. 真负例（True Negative，TN）：一个样本的真实类别为其他类，模型也预测为其他类

![image-20210525155039087](NLP-FQA_images/image-20210525155039087.png)

**精确率（Precision）**，也叫精度或查准率，

类别 𝑐 的查准率是所有预测为类别𝑐的样本中预测正确的比例:

![image-20210525155122489](NLP-FQA_images/image-20210525155122489.png)

召回率（Recall），也叫查全率，类别𝑐的查全率是所有真实标签为类别𝑐的样本中预测正确的比例：

![image-20210525155216428](NLP-FQA_images/image-20210525155216428.png)

F值（F Measure）是一个综合指标，为精确率和召回率的调和平均：

![image-20210525155253112](NLP-FQA_images/image-20210525155253112.png)

其中𝛽 用于平衡精确率和召回率的重要性，一般取值为1．𝛽 = 1时的F值称为F1值，是精确率和召回率的调和平均．



模型的评估指标

ROC坐标含义，AUC

比赛对应的评价指标? F1指标是怎么计算的? 了解对应的ROC曲线吗? 怎么计算对应的ROC曲线?

# 训练相关

BatchSize设定的大小都有什么关系吗？越大的BatchSize就越好吗？

层次softmax为什么? 怎么做的? 从根节点到叶子节点的过程是怎么样的?

导致梯度消失的原因？

Batch norm 公式， 与layer norm的区别？

# 数学+Codeing

实现余弦相似度计算

二叉树有了解吗，遍历有哪些（前中后层次）项目用过哪些数据结构（好像没有，都是用python做的）
动态规划了解吗，解释一下，填表是怎么做的（吧啦吧啦）

15.写一个二叉树的后序遍历，非递归，手写（双栈法，输出的时候有点小问题）

5、工程化的一些基础内容。Python中生成器、迭代器、多线程多进程。Linux中基本操作。模型部署Flask，多线程，docker化等。

61、了解python 的深浅拷贝吗？装饰器呢？

tf idf

64、欧氏距离、曼哈顿距离

65、tf-idf 怎么做的

66、手写一个tfidf

67、读取文件，计算TF-IDF

68、为什么tf-idf char 会对结果有提高？知道一些基本的词向量编码方法吗

10.快排知道吗？稳定不？（不稳定，因为会交换顺序）

13.最后做个笔试，用np实现一个LSTM，初始化权0.5（啊，能不能写伪代码？）不能写伪代码，要跑出来。（啊这。。。我好多np函数用法没背下来怎么办）你让我有点惊讶，居然没记下来这些常用方法（啊这个sigmoid怎么调用来着？）你自己写（啊好的好的，还好记得公式）

⑩最后问了几个计算机基础-组成原理的问题：计算机内存分为几个部分？局部变量放在哪里？由于手撕用java写的，问我刚刚写的定义的static变量又放在哪里？
1、算法部分：刷了《剑指offer》多遍+leetcode top interview。基本上面试中的问题都能够搞定。主要是解题、优化、时间空间复杂度等。

# 机器学习

GBDT的大概算法

KNN的大概算法

问了GBDT 和 XGBoost和LightGBM的大致改进.

介绍lightgbm。和决策树区别是什么？解决哪种误差？

LR，SVM，GBDT的原理

RNN、LSTM、GRU时序模型的特点

梯度弥散与梯度爆炸的原因与解决方案

激活函数的优缺点

什么是L2正则化，为什么L2正则化可以防止过拟合？L1正则化与L2正则化有什么区别和联系。

XGBoost

1、GBDT , XGBOOST, lightGBM 之间的异同。

2、XGBoost 的参数调优有哪些经验（工程能力）

3、XGBoost 的正则化是如何实现的（工程能力）

4、XGBoost的并行化部分是如何实现的（工程能力

5、Xgboost 优化点

6、决策树节点分裂时如何选择特征，写出Gini index 和Information Gain 的公式并举例

7、分类树和回归树的区别是什么？（理论基础）

8、与Random Forest 作比较，并以此介绍什么是模型的Bias 和Variance（理论基础）

9、xgboost 是用同一个模型吗？

11、各种聚类算法的优缺点划分：K-means 层次：AGNES 密度：DBSACN 模型：EM

12、Xgboost 优化点、LightGBM 与XGboost 的联系及区别，是都试过了么

13、如果选用一种其他的模型替代XGBoost，你会选用什么？

14、bagging、boosting、stacking 的异同。

# 其他

模型加速和优化的策略？

（讲了知识蒸馏）

为什么要用大模型训练了再采用小模型进行训练呢，反正不都是需要最后训练大模型么？（也不知道是我不懂还是面试官不懂了,反正我懵了）

模型蒸馏work的原因，具体实现方式

warm up 的的原理作用

动量优化

adam优化

量化、剪枝、参数共享等模型压缩手段

知识蒸馏的应用（同构蒸馏，模型压缩之类）

知识蒸馏的具体过程（从公式开始讲，teacher的soft prediction之类的，还有损失函数的构建）

特征工程是怎么做得？交叉特征是怎么做得的？

提升指标的一些trick 调参之前说过了，就说了加对抗训练，及实现细节

偏差、方差指什么？怎么减小

深层神经网络为什么不好训？除了梯度消失还有哪些原因？

autoencoder原理

tokenize的原理

ml相关模型介绍，原理，项目结果对比， 回归模型数据做归一化的必要性。

介绍一下TextCNN这些模型这样子.

讲一下线性模型和非线性模型的区别?

还有什么其他比较基础的Nlp技术

模型不收敛原因分析，解决方案（学习率，权重初始化，数据标注，样本不平衡问题等等

10、谈谈判别式模型和生成式模型？

17、神经网络中的梯度消失和梯度膨胀是什么，怎么解决?

18、激活函数的作用

19、如何检验过拟合，

20、CRF 的算法的解释。

21、介绍momentum，RMSprop 和Adam 等优化方式

26、attention 的原理，计算公式。

27、seq2seq 的缺点，如何改进？

41、在做NER 任务时，lstm 后面可以不用加CRF 吗？

42、BERT 和GPT 的区别

43、如何使用bert？

44、TextRank 原理？

45、聚类方式。K-means，层次聚类，DBSCAN...

47、介绍一下，一些基于BERT 的改进的模型。

48、写下bilstm+crf 的损失函数？损失函数中的max 有什么作用，为什么要减去这个max？具体的emission score 与transition score 是如何得到的？

49、推导LR 的损失函数与梯度下降更新参数（必须要手推一下，当时就没准备好）

51、Transformer 在GPT 和Bert 等词向量预训练模型中具体是怎么应用的？有什么变化？

52、NLP 特征抽取机制有哪些？(基于DN), 各有哪些优缺点

53、为什么是缩放点积，而不是点积模型？

54、为什么是双线性点积模型（经过线性变换Q ！=K）？

55、相较于加性模型，点积模型具备哪些优点？

56、用到了什么激活函数，怎么选择激活函数的使用。

59、自然语言中如何选择使用cnn 和rnn。

60、因为项目中使用到了macro-f1，所以问了marco-f1，为什么不用Precision、recall?

62、请简要介绍下SVM

63、请简要介绍下tensorflow 的计算图



69、glove 和fasttext 怎么训练的？

70、词向量如何训练的？（word2vector）

71、word2vector 为啥语义相近的词的词向量相近？怎么达到效果的？

72、softmax 和sigmod 区别？

bias和variance的区别
svm为什么要用核函数

15. COPA任务的复现原理
16. 文本相似度的对比
随机森林的介绍
F-alpha的公式
对残差网络的理解
micro和macro的区别
集成学习
高斯模型细讲
GMM
什么是核函数并举例
凸函数和拟凸函数
ELMO和LSTM
二叉搜索树讲解

介绍一下word confusion network

对比的baseline是什么，Word2vec,fasttext不能冻结emb不知道嘛? (知道，所以我后面用了transformer做对比实验)

PGN怎么解决OOV问题？

vocab_size为什么会减小？是不是理解成取topn个词频最大的？每一步的vocab怎么构建？

其他的copynet方式有了解么？（No）

EDA有了解么（只知道同义词替换这类的，没有详细了解）

GAN生成效果怎么样?为什么GAN很难训练?用过什么样子的GAN?
后续有没有考虑新的模型？是怎么考虑的？为什么要用TextCNN？

XGBoost具体的算法内容了解么?算法思想有哪些?

TextCNN的运作过程?
TextCNN的卷积核维度?
TextCNN的参数维度?

介绍一下GAN模型和XGBoost回归的做法，对于GAN训练困难是如何解决的。问了比赛团队分工的合作.模型融合是怎么做的。特征组合怎么选择的。是人工筛选的吗?

介绍一下XGBoost和LightGBM的算法的优缺点.

介绍GBDT算法的原理,介绍对应的LightGBM的对于XGBoost的一些优化?
简单问了一下XGBoost和LightGBM的算法的差异性。
开放题目：如何对爬虫收集到的文本数据判断对应的主题？
怎么做一个特征筛选？XGBoost是怎么预防过拟合的？

对于神经网络的本质是怎么理解的？（完全懵比，应该从一个问题的最优化的层面来介绍。然后给我介绍了从函数方面的理解。意思上比较像是引导我说对应的）

介绍一下TextCNN的模型。

了解强化学习吗？对强化学习的理解程度有多少？强化学习和普通神经网络间有什么差别？层次聚类是怎么做得？背景是什么？

还认识什么机器学习的模型? SVM中的松弛化是什么? 支持向量是什么?常用的核函数有哪些?怎么选取?
XGBoost和LightGBM的差别?

CNN和DNN的参数计算量差别?

问了解信息抽取吗?(撞我枪口上了吧!哈哈,拿着NER给它大讲了一通.)后来问用BERT+LSTM+CRF的训练数据大概要多少?会存在什么问题吗?

为什么选择模型的一个大致思路? 具体问了问我CRF(几个月前做的记不清了,大致讲了一下.)了不了解百度对BERT做的一些改进?(暗指ENRIE模型).个人在实际中遇到的一些困难,是如何解决掉的?

HMM的模型等, 问了HMM的改进模型有哪些?

HMM是一个独立考虑的模型,有没有对应的能考虑全局的模型?(????)

写代码的时候共享屏幕被看到的SVM的内容,又问了问SVM是怎样的模型.

了解过一些更优化的模型吗?如ALBERT,XLNet等.

手推LR,能不能从最小化损失函数的角度介绍介绍?
有没有了解过LR是服从什么的分布?
有没有了解损失函数的层面,有没有了解交叉熵.SVM过程大致介绍一下.SVM如何实现多分类

判断BST（自己定义TreeNode）
用randomInt(5)实现randomInt(7)，只用讲思路

介绍一下CNN，不同的卷积有了解吗？

平时用什么模型？LR、RF、SVM各自特点和应用场景？

介绍LightGBM。
6、GBDT原理，梯度下降过程，为什么是这个过程

5.介绍一下ernie模型

6还有什么其他比较基础的Nlp技术，我回答中包括了ner，于是问我目前比较流行的ner模型，我回答了bert\flat

7简单介绍一下flat及对于嵌套式语料的融合方式

8.code topK两种方式 （暴力，快排）

场景：有10万条短文本，需要打标签，应该怎么做？
我说编码后，用k-means聚类，还有别的嘛，不太了解了

3.假设一个情景题，微博创立初期，短文本，10w数据集，无监督，分类怎么做（我答的用label embedding做或者用embedding+聚类kmeans）

5.除了你说的方法，还有别的方法可以吗？U-learning听说过吗？（没听清？啥learing？这个真不知道，只听说过Q-learning）
6.kmeans的k怎么选择，这么做效果好吗？为什么？（k不知道，具体情况具体分析吧（编不出来了），效果不好）
7.那如果不用聚类，别的方法怎么做，提示：抽取关键字（关键字抽取可以用词与句子做相似度匹配）
8.按你说的做不具有通用性， 那如果是长文本，又怎么做，有什么区别（长文本用词频统计+停用词过滤）
9.你没回答出我想要的答案，因为一个微博可能属于多个类别（多类别无监督分类）


11.bagging、boosting知道吗？（知道，属于ensemble，叽里哇啦叽里哇啦）

分词知道嘛jieba说一说（知道，维特比算法巴拉巴拉）
3.分词种新词发现怎么做？（几里哇啦扯了一堆）你说的好像和我不是一个东西，新词发现有哪些常用方法（啊。。。不知道）
4.维特比算法中的DAG知道吗，怎么构建的（知道，扯了一堆）
5.所以DAG怎么构建你没有说（构建。。？不是根据统计信息得到的嘛？不知道）
6.看你熟悉维特比算法，HMM知道不（知道的...（西麻哒！！））
7.logP(O|A,B,pi)怎么变成只有ABpi的式子（啊这。。。扯了一堆，三个问题，参数估计，d-separated，动态规划（完全扯错了方向））



5.实体抽取怎么做的，有了解吗？（没有具体做，但是大概知道上NER这一块的东西，其他不清楚了）

10.树模型和深度学习的区别（只知道效果有差距，具体不太清楚）二者可以结合吗（不是很清楚）
11.树模型介绍一下（决策树、GBDT、XGBOOST、随机森林）核心思想介绍下（吧啦吧啦）
12.Adaboost和GBDT区别（Adaboost对错误给更高的权重，GBDT多个子树加起来（不知道说的对不对））
13.树模型和机器学习比如LR有啥区别（树模型有比较好的可以解释性（这个答案好像不是很满意，应该是稀疏性的问题））
14.稀疏性问题怎么解决（变成dense向量，说了一些分布式表示和one-hot的区别（不知道答偏了没有））
7.机器学习了解吗（了解），LR损失函数写一下（吧啦吧啦写了下，最后y写错了，应该放log外面）
8.FM了解吗？（不了解）Wide&Deep了解吗？（不了解，好像都是推荐系统的那边）
9.XGBOOT说一下（说了下和GBDT区别，之类的）为什么二阶导数好（信息更丰富）二阶导数也会更好的拟合（对对对）



若有个电子病历数据，你认为能识别多少种类型的实体。（场景题估计因人而异）
③.使用过的实体识别模型，大概介绍一下网络结构（从输入到输出）
④.HMM和CRF的区别，HMM是什么模型（判别式or生成式？）HMM的五个参数是什么，三个问题是什么。
⑤.GDBT如何抑制树的不断增长。
⑧说几个两年之内的前沿技术。
⑨平时逛过哪些有关NLP的论坛（这个问题被问过好多次）

​    2、深度学习基础部分：RNN、LSTM、Transformer、Bert、HMM、CRF等基本原理、细节内容，
​    3、项目部分：项目流程、模型算法、数据量、遇到问题解决办法。包括具体的优化、调参、工程化、数据提取存储、项目上线部署等。
BERT原理，各个细节原因，算法时间复杂度。怎么训练？数据量多少？如何评估？上线后响应速度？如何部署上线？多少个用户？多线程多进程等？数据不均衡的处理办法？BERT的变种等？
2、CRF原理，时间复杂度。与BERT组合怎么提升准确率。常见的评估标准哪些？怎么评估？怎么选择标准？
3、文本相似度用哪些方法？词向量、ELMO、BERT等？怎么提升比较的速度，多个中选择一个最相似的？
4、文本分类常用方法？评估指标？如果数据量大，但是标注的数据较少，怎么处理？