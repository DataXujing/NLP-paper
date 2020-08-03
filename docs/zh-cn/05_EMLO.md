
## ELMo(Embedding from Language Models)

NAACL 2018年 best paper, 艾伦人工智能研究中心

### 0.Abstract

本论文介绍了一种新的词向量表示方法(一种动态词向量)，（1）单词使用的复杂特征（如语法（syntax）和语义（semantics）），（2） 这些用法在不同的语言环境中是变化的（例如，建模多义词（polysemy））（动态词向量）。我们构建的词向量基于双向的语言模型(biLM),它是在一个大型文本语料库上预先训练的。我们证明这些表示可以很容易地添加到现有的模型和显著改善了六个具有挑战性的NLP问题的研究现状，包括问答(question answering)、语义分析(textual entailment)和情感分析(sentiment analysis)。 论文分析了ELMo的关键在于预训练模型的深层次结构， 允许下游模型（task)混合不同类型的半监督信号(基于base的vector加权平均)。

<!-- <div align=center>
    <img src="zh-cn/img/elmo/p1.png" /> 
</div> -->

### 1.Introduction

word2vector(Mikolov et al. 2013),GloVe等词向量在NLU中是非常重要的，但是会存在两个问题，（1）他们并没有考虑复杂的语法和语义，（2)多义词的问题（需要构建动态词向量），ELMo的提出解决了上述的两个问题。

我们的词表示不同于传统的词向量表示，我们每一个token的表示是他所在input的sentence的函数，我们使用一个大的语料库训练Bi-LSTM的语言模型。基于这个原因我们将该模型成为ELMo(Embedding from Language Models)表示。不同于其他之前的模型，ELMo词表示是深层的，是所有隐层的加权平均，而不是只是用LSTM的top的layer。浅层的特征表征语法，深层的特征表层语义。ELMo在6项NLP的测试任务中都表现很好，提升了20%的准确率。与**CoVe(McCann etal. 2017)**对比也是比CoVe表现要好。并且团队开源了算法： <http://allennlp.org/elmo>。


### 2.ELMo


ELMo基于语言模型的，确切的来说是一个 Bidirectional Language Models，也是一个 Bidirectional LSTM结构。我们要做的是给定一个含有$N$个tokens的序列

$$t_1,t_2,...,t_N$$

其前向表示为：

$$p(t_1,t_2,..,t_N)=\prod_{k=1}^{N}p(t_k|t_1,t_2,...,t_{k-1})$$

反向表示：

$$p(t_1,t_2,..,t_N)=\prod_{k=1}^{N}p(t_k|t_{k+},t_{k+2},...,t_N)$$

<div align=center>
    <img src="zh-cn/img/elmo/p1.jpg" /> 
</div>

从上面的联合概率来看是一个典型的语言模型，前向利用上文来预测下文，后向利用下文来预测上文。假设输入的token是$x_k^{LM}$，在每个位置$k$，每一层LSTM上都会输出相应的context-dependent的表征$\overrightarrow{h}_ {k,j}^{LM}$.这里$j=1,2,...,L$,$L$表示LSTM的层数。顶层的输出$\overleftarrow{h}_ {l,L}^{LM}$，通过softmax层来预测下一个$token_{k+1}$.

优化的目标：最大化对数前向和后向的似然概率

<div align=center>
    <img src="zh-cn/img/elmo/p2.png" /> 
</div>

<!-- \overleftarrow{ -->

所谓的ELMo不过是一些隐层向量的组合。都有哪些隐层向量呢？对于每个单词(token)$t_k$,对于L层的Bi-LSTM语言模型，一共有$2L+1$个表征(representations)

<div align=center>
    <img src="zh-cn/img/elmo/p3.png" /> 
</div>

其中$\overrightarrow{h}_ {k,j}^{LM},\overleftarrow{h}_ {k,j}^{LM}$表示第$k$个token的第$j$层Bi-LSTM的隐层向量。值得注意的是，每一层都有一个前向的LSTM和一个后向的LSTM,两者就简单的拼接(concat)起来的，也就是如果分别都是$m\times1$维的列向量，拼接玩之后就是$2m\times1$的列向量，就这么简单。

既然ELMo有这么多向量了，那怎么使用呢？最简单的方式是使用最顶层的LSTM输出，即$h_{k,L}^{LM}$,但是我们有更好的方法使用这些向量。


<div align=center>
    <img src="zh-cn/img/elmo/p4.png" /> 
</div>

对于每层向量都加入一个权重$s_j^{task}$,将每层的向量与权重相乘，然后再乘以一个scale的权重$\lambda^{task}$.每层LSTM输出或者每层LSTM学到的东西是不一样的，针对每个任务每层的向量的重要性也是不一样的，所以有L层LSTM,L+1个权重，加上scale的参数$\lambda^{task}$,一共L+2个权重。注意一下此处的权重个数，后面会用到。至于为什么要乘以$\lambda^{task}$，我们会在一下节看到，我们会将词向量与连一个向量再次评级人，所以此处有一个scale的系数。

笔者思考一个问题，为何不把L+1个向量一起拼接起来？这样网络可以学的更充分？是不是位数太高？



### 3.ELMo用于有监督的NLP任务

+ 情况1

(1).Freeze the weight of BiLM
(2).concatence $ELMo_k^{task}$和input的将单词/词条的表征$x_k$，形成增强表示$[x_k;ELMo_k^{task}]$,并添加到RNN中。

+ 情况2

对于一些任务(如SNLI，SQuAD)，通过引入了另一组输出的线性权值，将$h_k$替换为：$[h_k; ELMo_k^{task}]$，这样可以observe further improvements.

+ 情况3

剩下的superveised model未变，这些additions可以在更复杂的神经模型上下文中发生
例如：`biLSTMs + bi-attention layer`， 或 一个放在Bi-LSTM之上的聚类模型.

在ELMo中加入适量的dropout是有益的,一些情况下添加$\lambda||w||^2$(L2正则化）到Loss中来regularize ELMo weights也是有益的
这对ELMo权重施加了一个归纳偏差，使其接近于所有biLM层的平均值。

### 4.预训练的双向语言模型架构

论文的作者有预训练好的ELMo模型，映射层（单词到word embedding）使用的Jozefowicz et al.(2016)的CNN-BIG-LSTM[5]，即输入为512维的列向量。同时LSTM的层数L，最终使用的是2，即L=2。LSTM的单元数是4096。每个LSTM的输出也是512维列向量。每层LSTM（含前、向后向两个）的单元个数是4096个（从上述可知公式$4m\times 2=4\times 512\times 2=4096$）。也就是每层的单个LSTM的输入是512维，输出也是512维。

一旦模型预训练完成，便可以用于NLP其他任务。在一些领域，可以对biLM（双向LSTM语言模型）进行微调，对任务的表现会有所提高，这种可以认为是一种迁移学习（transfer learning）。

### 5.ELMo使用方法总结及效果展示

对于预训练好的Bi-LSTM语言模型，我们可以送入一段话，然后模型会得到如下图的向量，

<div align=center>
    <img src="zh-cn/img/elmo/p3.png" /> 
</div>


然后我们加上一定的权重（可训练）即可得到如下图的ELMo向量，

<div align=center>
    <img src="zh-cn/img/elmo/p4.png" /> 
</div>


最终将ELMo向量与$x_k$拼接作为单词的特征，用于后续的处理。

对于部分任务，可以对双向lstm语言模型微调，可能有所提升。

至于ELMo的效果，可以参考下图

<div align=center>
    <img src="zh-cn/img/elmo/p5.png" /> 
</div>

上表显示了ELMo在6个不同的NLP任务中的性能。在考虑的每一项任务中，简单的添加ELMo就可以得到一个新的更精确的结果，与强大的基准相比，相对误差见笑了6-20%，这是跨多种模型体系结构和语言理解任务的非常普遍的结果。如第一项斯坦福问答数据集SQuAD，包含10万条来源与人群的问答对，答案是一个给定的维基百科段落。在将ELMo添加到基准模型后，测试集的性能提高了4.7%，从81.1%提高到85.8%,比基准模型降低了24.9%的相对误差，整体单一模型的性能提高了1.4%.

### 6.ELMo学到了什么

**ELMo到底学到了什么呢？我们前文提到的多义词问题解决了吗？**

可以观察下图，可以看到，加入ELMo之后，可以明显将`play`的两种含义区分出来，而GLoVe并不能。所以答案很明显。

<div align=center>
    <img src="zh-cn/img/elmo/p6.png" /> 
</div>

**Word sense disambiguation（词义消歧）**

作者是通过实验证明的，如下图。biLM表示我们的模型。第一层，第二层分别使用的结果显示，越高层，对语义理解越好，表示对词义消歧做的越好。这表明，越高层，越能捕获词意信息。

<div align=center>
    <img src="zh-cn/img/elmo/p7.png" /> 
</div>

**POS tagging（词性标注）**

这是另一个任务的实验了，如下图，第一层效果好于第二层。表明，低层的更能学到词的句法信息和词性信息。

<div align=center>
    <img src="zh-cn/img/elmo/p8.png" /> 
</div>

总体而言，biLM每层学到的东西是不一样的，所以将他们叠加起来，对任务有较好的的提升。

### 7.ELMo的缺点


前文提了这么多ELMo的优点，现在说一说缺点。这些缺点笔者是搬运<https://zhuanlan.zhihu.com/p/51679783>的观点。<https://zhuanlan.zhihu.com/p/51679783>的观点是站在现在的时间点上（Bert已发布）看的，他的观点如下：

那么站在现在这个时间节点看，ELMo 有什么值得改进的缺点呢？首先，一个非常明显的缺点在特征抽取器选择方面，ELMo 使用了 LSTM 而不是新贵 Transformer，Transformer 是谷歌在 2017 年做机器翻译任务的“Attention is all you need”的论文中提出的，引起了相当大的反响，很多研究已经证明了 Transformer 提取特征的能力是要远强于 LSTM 的。如果 ELMo 采取 Transformer 作为特征提取器，那么估计 Bert 的反响远不如现在的这种火爆场面。另外一点，ELMo 采取双向拼接这种融合特征的能力可能比 Bert 一体化的融合特征方式弱，但是，这只是一种从道理推断产生的怀疑，目前并没有具体实验说明这一点。

### 8.ELMo简单上手

既然ELMo有这么有用，该怎么使用呢？这里介绍一下简单的使用方法。有三种方法可以使用预训练好的ELMo模型。

+ ELMo官方allenNLP发布的基于pytorch实现的版本<https://github.com/allenai/allennlp>；
+ ELMo官方发布的基于tensorflow实现的版本<https://github.com/allenai/bilm-tf>；
+ tensorflow-hub中google基于tensorflow实现的elmo的版本<https://tfhub.dev/google/elmo/2>

本节内容介绍第三个版本。

先简单介绍下tensorflow-hub，hub类似于github的hub，tensorflow-hub的目标就是讲机器学习的算法，包含数据、训练结果、参数等都保存下来，类似于github一样，拿来就可以直接用。所有人都可以在这里提交自己的模型及数据、参数等。这里实现的ELMo是google官方实现并预训练好的模型。有人基于此模型+keras写的博客及代码教程大家可以参考下<https://link.zhihu.com/?target=https%3A//towardsdatascience.com/elmo-embeddings-in-keras-with-tensorflow-hub-7eb6f0145440>,<https://link.zhihu.com/?target=https%3A//github.com/strongio/keras-elmo/blob/master/Elmo%2520Keras.ipynb>,此代码使用的google的ELMo的第一个版本，目前最新的是第二个版本。

下面看代码的简单上手使用，大家可能需要先安装tensorflow_hub。

```
import tensorflow_hub as hub

# 加载模型
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
# 输入的数据集
texts = ["the cat is on the mat", "dogs are in the fog"]
embeddings = elmo(
texts,
signature="default",
as_dict=True)["default"]
```

上述代码中，`hub.Module`加载模型，第一次会非常慢，因为要下载模型，甚至可能要科学上网。该模型是训练好的模型，也就是LSTM中的参数都是固定的。这里的`trainable=True`是指第2节中提到的4个权重参数可以训练。`texts`是输入数据集的格式，也有另一种输入格式，代码如下。`signature`为`defaul`t时，输入就是上面的代码，`signature`为`tokens`时，就是下面的方式输入。注意最后一行的中括号里的`default`，表示输出的内容。这个`default`位置有五个参数可以选，分别为：

1. word_emb，表示word embedding，这个就纯粹相当于我们之前提到的lstm输入的位置的word embedding，维数是[batch_size, max_length, 512]，batch_size表示样本的个数，max_length是样本中tokens的个数的最大值，最后是每个word embedding是512维。
2. lstm_outputs1，第一层双向lstm的输出，维数是[batch_size, max_length, 1024]。
3. lstm_outputs2，第二层双向lstm的输出，维数是[batch_size, max_length, 1024]。4. elmo，输入层及两个输出层，三层乘以权重。其中权重是可以训练的，如第2节所讲。维数是[batch_size, max_length, 1024]。5.default，a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024]。 所以可以看到，要想训练权重，要使用elmo这个参数。

```
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

# 另一种方式输入数据
tokens_input = [["the", "cat", "is", "on", "the", "mat"],
["dogs", "are", "in", "the", "fog", ""]]
# 长度，表示tokens_input第一行6一个有效，第二行5个有效
tokens_length = [6, 5]
# 生成elmo embedding
embeddings = elmo(
inputs={
"tokens": tokens_input,
"sequence_len": tokens_length
},
signature="tokens",
as_dict=True)["default"]
```

上面生成的embedding，想变成numpy向量，可以使用下面的代码。

```
from tensorflow.python.keras import backend as K

sess = K.get_session()
array = sess.run(embeddings)

```

至此，关于ELMo的所有内容已经完毕了。更多的使用，还需要再探索。谢谢大家。