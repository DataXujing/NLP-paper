
本节我们介绍BERT，BERT是2018年由Google发表的基于Transformer encoder的语言模型。在介绍BERT之前，我们将介绍在NLP领域中几个常用的Toy Data和在BERT中使用的激活函数`gelu`激活函数。

## NLP中常用的Toy Data

<!-- https://ana.cachopo.org/datasets-for-single-label-text-categorization -->
<!-- https://blog.csdn.net/Gavin__Zhou/article/details/78286379?utm_source=blogxgwz6 -->
<!-- https://zhuanlan.zhihu.com/p/46834868 -->
<!-- https://blog.csdn.net/enohtzvqijxo00atz3y8/article/details/8016306 -->
<!-- https://github.com/awesomedata/awesome-public-datasets -->


可以参考如下github:

+ https://github.com/awesomedata/awesome-public-datasets
+ https://github.com/niderhoff/nlp-datasets

------


## 激活函数gelu

Gaussian Error Linerar Units: <https://arxiv.org/abs/1606.08415>

最近在看BERT源码，发现里边的激活函数不是Relu等常见的函数，是一个新的激活函数GELUs, 这里记录分析一下该激活函数的特点。

不管其他领域的鄙视链，在激活函数领域，大家公式的鄙视链应该是：`ElUs > Relu > Sigmoid` ，这些激活函数都有自身的缺陷， sigmoid容易饱和，ElUs与Relu缺乏随机因素。

在神经网络的建模过程中，模型很重要的性质就是非线性，同时为了模型泛化能力，需要加入随机正则，例如dropout(随机置一些输出为0,其实也是一种变相的随机非线性激活)， 而随机正则与非线性激活是分开的两个事情， 而其实模型的输入是由非线性激活与随机正则两者共同决定的。

GELUs正是在激活中引入了随机正则的思想，是一种对神经元输入的概率描述，直观上更符合自然的认识，同时实验效果要比Relu与ELUs都要好。

GELUs其实是 dropout、zoneout、Relus的综合，GELUs对于输入乘以一个0,1组成的mask，而该mask的生成则是依概率随机的依赖于输入。假设输入为`X`, mask为`m`，则`m`服从一个伯努利分布`(Φ(x), Φ(x)=P(X<=x),X服从标准正太分布)`，这么选择是因为神经元的输入趋向于正太分布，这么设定使得当输入x减小的时候，输入会有一个更高的概率被dropout掉，这样的激活变换就会随机依赖于输入了。

数学表达如下：

$$GELU(x)=xP(X<=x)=x\Phi(x)$$

这里`Φ(x)`是正太分布的概率函数，可以简单采用正太分布`N(0,1)`, 要是觉得不刺激当然可以使用参数化的正太分布`N(μ,σ)`, 然后通过训练得到`μ,σ`。

对于假设为标准正太分布的`GELU(x)`, 论文中提供了近似计算的数学公式，如下：

$$GELU(x)=0.5x(1=\tanh[\sqrt{(2/\pi)}(x+0.044715x^3)])$$

翻看BERT源码给出的GELU代码表示如下：
```python
def gelu(input_tensor):
	cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
	return input_tesnsor*cdf
```
感觉BERT源码中的近似计算更简单，具体怎么近似的，我猜不出来。

下面贴一些论文的实验图，就是证明GELU学习更快且更好：

<div align=center>
    <img src="zh-cn/img/bert/gelu/p1.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/bert/gelu/p2.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/bert/gelu/p3.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/bert/gelu/p4.png" /> 
</div>


------

## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

BERT：预训练的深度双向 Transformer 语言模型

Jacob Devlin；Ming-Wei Chang；Kenton Lee；Kristina Toutanova
Google AI Language

{jacobdevlin,mingweichang,kentonl,kristout}@google.com


### 摘要
我们提出了一种新的称为 BERT 的语言表示模型，BERT 代表来自 Transformer 的双向编码器表示（**B**idirectional **E**ncoder **R**epresentations from **T**ransformers）。不同于最近的语言表示模型（[Peters et al., 2018](https://arxiv.org/abs/1802.05365v2)，[Radford et al., 2018](https://blog.openai.com/language-unsupervised/)）， BERT 旨在通过联合调节所有层中的左右上下文来预训练深度双向表示。因此，只需要一个额外的输出层，就可以对预训练的 BERT 表示进行微调，从而为广泛的任务（比如回答问题和语言推断任务）创建最先进的模型，而无需对特定于任务进行大量模型结构的修改。

BERT 的概念很简单，但实验效果很强大。它刷新了 11 个 NLP 任务的当前最优结果，包括将 GLUE 基准提升至 80.4%（7.6% 的绝对改进）、将 MultiNLI 的准确率提高到 86.7%（5.6% 的绝对改进），以及将 SQuAD v1.1 的问答测试 F1 得分提高至 93.2 分（提高 1.5 分）——比人类表现还高出 2 分。

### 1. 介绍
语言模型预训练可以显著提高许多自然语言处理任务的效果（[Dai and Le, 2015](http://papers.nips.cc/paper/5949-semi-supervised-sequence-learning)；[Peters et al., 2018](https://arxiv.org/abs/1802.05365v2)；[Radford et al., 2018](https://blog.openai.com/language-unsupervised/)；[Howard and Ruder, 2018](https://arxiv.org/abs/1801.06146v5)）。这些任务包括句子级任务，如自然语言推理（[Bow-man et al., 2015](https://arxiv.org/abs/1508.05326v1)；[Williams et al., 2018](https://arxiv.org/abs/1704.05426v4)）和释义（[Dolan and Brockett, 2005](https://www.researchgate.net/publication/228613673_Automatically_constructing_a_corpus_of_sentential_paraphrases)），目的是通过对句子的整体分析来预测句子之间的关系，以及标记级任务，如命名实体识别（[Tjong Kim Sang and De Meulder, 2003](http://www.oalib.com/paper/4018980)）和 SQuAD 问答（[Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250v3)），模型需要在标记级生成细粒度的输出。

现有的两种方法可以将预训练好的语言模型表示应用到下游任务中：基于特征的和微调。基于特征的方法，如 ELMo （[Peters et al., 2018](https://arxiv.org/abs/1802.05365v2))，使用特定于任务的模型结构，其中包含预训练的表示作为附加特特征。微调方法，如生成预训练 Transformer  (OpenAI GPT) （[Radford et al., 2018](https://blog.openai.com/language-unsupervised/)）模型，然后引入最小的特定于任务的参数，并通过简单地微调预训练模型的参数对下游任务进行训练。在之前的工作中，两种方法在预训练任务中都具有相同的目标函数，即使用单向的语言模型来学习通用的语言表达。

我们认为，当前的技术严重地限制了预训练表示的效果，特别是对于微调方法。主要的局限性是标准语言模型是单向的，这就限制了可以在预训练期间可以使用的模型结构的选择。例如，在 OpenAI GPT 中，作者使用了从左到右的模型结构，其中每个标记只能关注 Transformer 的自注意层中该标记前面的标记（[Williams et al., 2018](https://arxiv.org/abs/1704.05426v4)）。这些限制对于句子级别的任务来说是次优的（还可以接受），但当把基于微调的方法用来处理标记级别的任务（如 SQuAD 问答）时可能会造成不良的影响（[Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250v3)），因为在标记级别的任务下，从两个方向分析上下文是至关重要的。

在本文中，我们通过提出 BERT 改进了基于微调的方法：来自 Transformer 的双向编码器表示。受完形填空任务的启发，BERT 通过提出一个新的预训练任务来解决前面提到的单向约束：“遮蔽语言模型”（MLM masked language model）（[Tay-lor, 1953](https://www.researchgate.net/publication/232539913_Cloze_Procedure_A_New_Tool_For_Measuring_Readability)）。遮蔽语言模型从输入中随机遮蔽一些标记，目的是仅根据被遮蔽标记的上下文来预测它对应的原始词汇的 id。与从左到右的语言模型预训练不同，MLM 目标允许表示融合左右上下文，这允许我们预训练一个深层双向 Transformer。除了遮蔽语言模型之外，我们还提出了一个联合预训练文本对来进行“下一个句子预测”的任务。

本文的贡献如下：
+ 我们论证了双向预训练对语言表征的重要性。与 [Radford et al., 2018](https://blog.openai.com/language-unsupervised/) 使用单向语言模型进行预训练不同，BERT 使用遮蔽语言模型来实现预训练深层双向表示。这也与 [Peters et al., 2018](https://arxiv.org/abs/1802.05365v2) 的研究形成了对比，他们使用了一个由左到右和由右到左的独立训练语言模型的浅层连接。
+ 我们表明，预训练的表示消除了许多特定于任务的高度工程化的的模型结构的需求。BERT 是第一个基于微调的表示模型，它在大量的句子级和标记级任务上实现了最先进的性能，优于许多特定于任务的结构的模型。
+ BERT 为 11 个 NLP 任务提供了最先进的技术。我们还进行大量的消融研究，证明了我们模型的双向本质是最重要的新贡献。代码和预训练模型将可在 [goo.gl/language/bert](https://github.com/google-research/bert) 获取。

### 2 相关工作
预训练通用语言表示有很长的历史，我们将在本节简要回顾最流行的方法。

#### 2.1 基于特征的方法
几十年来，学习广泛适用的词语表示一直是一个活跃的研究领域，包括非神经网络学领域（[Brown et al., 1992](https://dl.acm.org/citation.cfm?id=176316);[](http://academictorrents.com/details/f4470eb8bc3a6f697df61bde319fd56e3a9d6733);[Blitzer et al., 2006](https://dl.acm.org/citation.cfm?id=1610094)）和神经网络领域（[Collobert and Weston, 2008](https://www.researchgate.net/publication/200044432_A_Unified_Architecture_for_Natural_Language_Processing)；[Mikolov et al., 2013](https://arxiv.org/abs/1310.4546v1)；[Pennington et al., 2014](http://www.aclweb.org/anthology/D14-1162)）方法。经过预训练的词嵌入被认为是现代 NLP 系统的一个不可分割的部分，词嵌入提供了比从头开始学习的显著改进（[Turian et al., 2010](https://www.researchgate.net/publication/220873681_Word_Representations_A_Simple_and_General_Method_for_Semi-Supervised_Learning)）。

这些方法已被推广到更粗的粒度，如句子嵌入（[Kiros et al., 2015](https://arxiv.org/abs/1506.06726v1)；[Logeswaran and Lee, 2018](https://arxiv.org/abs/1803.02893v1)）或段落嵌入（[Le and Mikolov, 2014](https://arxiv.org/abs/1405.4053v2)）。与传统的单词嵌入一样，这些学习到的表示通常也用作下游模型的输入特征。

ELMo（[Peters et al., 2017](https://arxiv.org/abs/1705.00108v1)）从不同的维度对传统的词嵌入研究进行了概括。他们建议从语言模型中提取上下文敏感的特征。在将上下文嵌入与特定于任务的架构集成时，ELMo 为几个主要的 NLP 标准提供了最先进的技术（[Peters et al., 2018](https://arxiv.org/abs/1802.05365v2))，包括在 SQuAD 上的问答（[Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250v3)），情感分析（[Socher et al., 2013](https://nlp.stanford.edu/sentiment/)），和命名实体识别（[jong Kim Sang and De Meul-der, 2003](http://www.oalib.com/paper/4018980)）。

#### 2.2 基于微调的方法
语言模型迁移学习（LMs）的一个最新趋势是，在对受监督的下游任务的模型进行微调之前，先对 LM 目标上的一些模型构造进行预训练（[Dai and Le, 2015](http://papers.nips.cc/paper/5949-semi-supervised-sequence-learning)；[Howard and Ruder, 2018](https://arxiv.org/abs/1801.06146v5)；[Radford et al., 2018](https://blog.openai.com/language-unsupervised/)）。这些方法的优点是只有很少的参数需要从头开始学习。至少部分得益于这一优势，OpenAI GPT （[Radford et al., 2018](https://blog.openai.com/language-unsupervised/)）在 GLUE 基准测试的许多句子级任务上取得了此前最先进的结果（[Wang et al.(2018)](https://arxiv.org/abs/1804.07461v2))。

#### 2.3 从有监督的数据中迁移学习
虽然无监督预训练的优点是可用的数据量几乎是无限的，但也有研究表明，从具有大数据集的监督任务中可以进行有效的迁移，如自然语言推理（[Con-neau et al., 2017](https://www.aclweb.org/anthology/D17-1070)）和机器翻译（[McCann et al., 2017](https://einstein.ai/static/images/pages/research/cove/McCann2017LearnedIT.pdf)）。在NLP之外，计算机视觉研究也证明了从大型预训练模型中进行迁移学习的重要性，有一个有效的方法可以微调在 ImageNet 上预训练的模型（[Deng et al., 2009](https://ieeexplore.ieee.org/document/5206848)；[Yosinski et al., 2014](https://arxiv.org/abs/1411.1792v1)）

### 3 BERT
本节将介绍 BERT 及其具体实现。首先介绍了 BERT 模型结构和输入表示。然后我们在 3.3 节介绍本文的核心创新——预训练任务。在 3.4 和 3.5 节中分别详细介绍了预训练过程和微调模型过程。最后，在 3.6 节中讨论了 BERT 和 OpenAI GPT 之间的区别。

#### 3.1 模型结构
BERT 的模型结构是一个基于 [Vaswani et al.(2017)](https://arxiv.org/abs/1706.03762v5)  描述的原始实现的多层双向 Transformer 编码器，并且 Transformer 编码器发布在 [tensor2tensor](https://github.com/tensorflow/tensor2tensor) 代码库中。由于最近 Transformer 的使用已经非常普遍，而且我们的实现与最初的实现实际上是相同的，所以我们将省略对模型结构的详尽的背景描述，并向读者推荐 [Vaswani et al.(2017)](https://arxiv.org/abs/1706.03762v5) 以及优秀的指南，如“[带注释的 Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)”。

在这项工作中，我们表示层的数量(即，Transformer 块)为 $L$，隐藏尺寸为 $H$，自注意头的个数为 $A$。在所有例子中，我们将前馈/过滤器的大小设置为 $4H$，即当 $H = 768$ 时是 $3072$；当 $H = 1024$ 是 $4096$。我们主要分析两个模型大小的结果:
+ $BERT_{BASE}: L=12, H=768, A=12, Total Parameters=110M$
+ $BERT_{LARGE}: L=24, H=1024, A=16, Total Parameters=340M$

为了方便比较，$BERT_{BASE}$ 选择了与 OpenAI GPT 一样的模型大小。然而，重要的是，BERT Transformer 使用的是双向的自注意力，而 GPT Transformer 使用的是受限的自注意力，每个标记只能关注其左边的语境。我们注意到，在文献中，双向 Transformer 通常被称为“Transformer 编码器”，而只有标记左侧语境的版本由于可以用于文本生成而被重新定义为“Transformer 解码器”。BERT、OpenAI GPT 和 ELMo 之间的比较如图 1 所示。

<div align=center>
    <img src="zh-cn/img/bert/figure_1.png" /> 
</div>

<!-- ![](zh-cn/img/bert/figure_1.png) -->

> 图 1：预训练模型结构的不同。BERT 使用双向 Transformer。OpenAI GPT 使用 从左到右的 Transformer。ELMo 使用独立训练的从左到右和从右到左的 LSTM 的连接来为下游任务生成特征。其中，只有 BERT 表示在所有层中同时受到左右语境的制约。

#### 3.2 输入表示
我们的输入表示能够在一个标记序列中清楚地表示单个文本句子或一对文本句子(例如，[Question, Answer])。（注释：在整个工作中，“句子”可以是连续的任意跨度的文本，而不是实际语言意义上的句子。“序列”是指输入到 BERT 的标记序列，它可以是单个句子，也可以是两个句子组合在一起。）通过把给定标记对应的标记嵌入、句子嵌入和位置嵌入求和来构造其输入表示。图 2 给出了输入表示的可视化表示。
细节是:
+ 我们使用含 3 万个标记词语的 WordPiece 嵌入（[Wu et al., 2016](https://arxiv.org/abs/1609.08144v2)）。我们用 ## 表示拆分的单词片段。
+ 我们使用学习到的位置嵌入，支持的序列长度最长可达 512 个标记。
+ 每个序列的第一个标记始终是特殊分类嵌入（[CLS]）。该特殊标记对应的最终隐藏状态（即，Transformer 的输出）被用作分类任务中该序列的总表示。对于非分类任务，这个最终隐藏状态将被忽略。
+ 句子对被打包在一起形成一个单独的序列。我们用两种方法区分这些句子。方法一，我们用一个特殊标记（[SEP]）将它们分开。方法二，我们给第一个句子的每个标记添加一个可训练的句子 A 嵌入，给第二个句子的每个标记添加一个可训练的句子 B 嵌入。
+ 对于单句输入，我们只使用句子 A 嵌入。

<div align=center>
    <img src="zh-cn/img/bert/figure_2.png" /> 
</div>

<!-- ![](zh-cn/img/bert/figure_2.png) -->

> 图 2：BERT 的输入表示。输入嵌入是标记嵌入（词嵌入）、句子嵌入和位置嵌入的总和。

##### 3.3.1 任务一#：遮蔽语言模型
直觉上，我们有理由相信，深度双向模型严格来说比从左到右模型或从左到右模型结合从右到左模型的浅层连接更强大。不幸的是，标准条件语言模型只能从左到右或从右到左进行训练，因为双向条件作用将允许每个单词在多层上下文中间接地“看到自己”。

为了训练深度双向表示，我们采用了一种简单的方法，即随机遮蔽一定比例的输入标记，然后仅预测那些被遮蔽的标记。我们将这个过程称为“遮蔽语言模型”（MLM），尽管在文献中它通常被称为完形填词任务（[Taylor, 1953](https://www.researchgate.net/publication/232539913_Cloze_Procedure_A_New_Tool_For_Measuring_Readability)）。在这种情况下，就像在标准语言模型中一样，与遮蔽标记相对应的最终隐藏向量被输入到与词汇表对应的输出 softmax 中（也就是要把被遮蔽的标记对应为词汇表中的一个词语）。在我们所有的实验中，我们在每个序列中随机遮蔽 15% 的标记。与去噪的自动编码器（[Vincent et al., 2008](https://www.researchgate.net/publication/221346269_Extracting_and_composing_robust_features_with_denoising_autoencoders)）不同的是，我们只是让模型预测被遮蔽的标记，而不是要求模型重建整个输入。

虽然这确实允许我们获得一个双向预训练模型，但这种方法有两个缺点。第一个缺点是，我们在预训练和微调之间造成了不匹配，因为 [MASK] 标记在微调期间从未出现过。为了缓和这种情况，我们并不总是用真的用 [MASK] 标记替换被选择的单词。而是，训练数据生成器随机选择 15% 的标记，例如，在my dog is hairy 这句话中，它选择 hairy。然后执行以下步骤:
+ 数据生成不会总是用 [MASK] 替换被选择的单词，而是执行以下操作:
+ 80% 的情况下：用 [MASK] 替换被选择的单词，例如，my dog is hairy → my dog is [MASK]
+ 10% 的情况下：用一个随机单词替换被选择的单词，例如，my dog is hairy → my dog is apple
+ 10% 的情况下：保持被选择的单词不变，例如，my dog is hairy → my dog is hairy。这样做的目的是使表示偏向于实际观察到的词。

Transformer 编码器不知道它将被要求预测哪些单词，或者哪些单词已经被随机单词替换，因此它被迫保持每个输入标记的分布的上下文表示。另外，因为随机替换只发生在 1.5% 的标记（即，15% 的 10%）这似乎不会损害模型的语言理解能力。

第二个缺点是，使用 Transformer 的每批次数据中只有 15% 的标记被预测，这意味着模型可能需要更多的预训练步骤来收敛。在 5.3 节中，我们证明了 Transformer 确实比从左到右的模型（预测每个标记）稍微慢一点，但是 Transformer 模型的实验效果远远超过了它增加的预训练模型的成本。

##### 3.3.2 任务2#：下一句预测
许多重要的下游任务，如问题回答（QA）和自然语言推理（NLI），都是建立在理解两个文本句子之间的关系的基础上的，而这并不是语言建模直接捕捉到的。为了训练一个理解句子关系的模型，我们预训练了一个下一句预测的二元分类任务，这个任务可以从任何单语语料库中简单地归纳出来。具体来说，在为每个训练前的例子选择句子 A 和 B 时，50% 的情况下 B 是真的在 A 后面的下一个句子，50% 的情况下是来自语料库的随机句子。比如说:

<!-- ![](zh-cn/img/bert/3_3_2_1.png) -->
<div align=center>
    <img src="zh-cn/img/bert/3_3_2_1.png" /> 
</div>

我们完全随机选择不是下一句的句子，最终的预训练模型在这个任务中达到了 97%-98% 的准确率。尽管这个任务很简单，但是我们在 5.1 节中展示了针对此任务的预训练对 QA 和 NLI 都非常有益。

#### 3.4 预训练过程
预训练过程大体上遵循以往文献中语言模型预训练过程。对于预训练语料库，我们使用 BooksCorpus（800M 单词）（[Zhu et al., 2015](https://arxiv.org/abs/1506.06724v1)）和英语维基百科（2,500M 单词）。对于维基百科，我们只提取文本段落，而忽略列表、表格和标题。为了提取长的连续序列，使用文档级别的语料库，而不是使用像 Billion Word Benchmark （[Chelba et al., 2013](https://arxiv.org/abs/1312.3005v3)）那样使用打乱顺序的句子级别语料库是至关重要的。

为了生成每个训练输入序列，我们从语料库中采样两段文本，我们将其称为“句子”，尽管它们通常比单个句子长得多（但也可以短一些）。第一个句子添加 A 嵌入，第二个句子添加 B 嵌入。50% 的情况下 B 确实是 A 后面的实际下一句，50% 的情况下它是随机选取的一个的句子，这是为“下一句预测”任务所做的。两句话合起来的长度要小于等于 512 个标记。语言模型遮蔽过程是在使用 WordPiece 序列化句子后，以均匀的 15% 的概率遮蔽标记，不考虑部分词片的影响（那些含有被 WordPiece 拆分，以##为前缀的标记）。

我们使用 256 个序列（256 个序列 * 512 个标记= 128,000 个标记/批次）的批大小进行 1,000,000 步的训练，这大约是在 33 亿词的语料库中训练 40 个周期。我们用Adam 优化算法并设置其学习率为 $1e-4$，$β1 = 0.9,β2 = 0.999$，$L2$ 的权重衰减是 0.01，并且在前 10000 步学习率热身（learning rate warmup），然后学习率开始线性衰减。我们在所有层上使用 0.1 概率的 dropout。像 OpenAI GPT 一样，我们使用 gelu 激活（[Hendrycks and Gimpel, 2016](https://arxiv.org/abs/1606.08415v3)）而不是标准 relu。训练损失是遮蔽语言模型似然值与下一句预测似然值的平均值。

在 4 块 [Cloud TPU](https://cloudplatform.googleblog.com/2018/06/Cloud-TPU-now-offers-preemptible-pricing-and-global-availability.html)（共含有 16 块 TPU）上训练 $BERT_{BASE}$。在 16 块 Cloud TPU（共含有 64 块 TPU）训练 $BERT_{LARGE}$。每次训练前需要 4 天的时间。

#### 3.5 微调过程
对于序列级别的分类任务，BERT 微调非常简单。为了获得输入序列的固定维度的表示，我们取特殊标记（[CLS]）构造相关的嵌入对应的最终的隐藏状态(即，为 Transformer 的输出)的池化后输出。我们把这个向量表示为 $C \in \mathbb{R}^H$，在微调期间唯一需要的新增加的参数是分类层的参数矩阵 $W \in \mathbb{R}^{K \times H}$，其中 $K$ 是要分类标签的数量。分类标签的概率$P \in \mathbb{R}^K$ 由一个标准的 softmax 来计算，$P=softmax(CW^T)$。对 BERT 的参数矩阵 $W$ 的所有参数进行了联合微调，使正确标签的对数概率最大化。对于区间级和标记级预测任务，必须以特定于任务的方式稍微修改上述过程。具体过程见第 4 节的相关内容。

对于微调，除了批量大小、学习率和训练次数外，大多数模型超参数与预训练期间相同。Dropout 概率总是使用 0.1。最优超参数值是特定于任务的，但我们发现以下可能值的范围可以很好地在所有任务中工作:

+ Batch size: 16, 32
+	Learning rate (Adam): 5e-5, 3e-5, 2e-5
+	Number of epochs: 3, 4

我们还观察到大数据集（例如 100k+ 标记的训练集）对超参数选择的敏感性远远低于小数据集。微调通常非常快，因此只需对上述参数进行完全搜索，并选择在验证集上性能最好的模型即可。

#### 3.6 BERT 和 OpenAI GPT 的比较
在现有预训练方法中，与 BERT 最相似的是 OpenAI GPT，它在一个大的文本语料库中训练从左到右的 Transformer 语言模型。事实上，BERT 中的许多设计决策都是有意选择尽可能接近 GPT 的，这样两种方法就可以更加直接地进行比较。我们工作的核心论点是，在 3.3 节中提出的两项新的预训练语言模型任务占了实验效果改进的大部分，但是我们注意到 BERT 和 GPT 在如何训练方面还有其他几个不同之处:

+ GPT 是在 BooksCorpus（800M 词）上训练出来的；BERT 是在 BooksCor-pus（800M 词）和 Wikipedia（2,500M 词）上训练出来的。
+ GPT 仅在微调时使用句子分隔符（[SEP]）和分类标记（[CLS]）；BERT 在预训练时使用 [SEP]， [CLS] 和 A/B 句嵌入。
+ GPT 在每批次含 32,000 词上训练了 1M 步；BERT 在每批次含 128,000 词上训练了 1M 步。
+ GPT 在所有微调实验中学习速率均为 5e-5；BERT 选择特定于任务的在验证集中表现最好的微调学习率。

为了分清楚这些差异的带来的影响，我们在 5.1 节中的进行每一种差异的消融实验表明，大多数的实验效果的改善实际上来自新的预训练任务（遮蔽语言模型和下一句预测任务）。

<div align=center>
    <img src="zh-cn/img/bert/figure_3.png" /> 
</div>
<!-- ![](zh-cn/img/bert/figure_3.png) -->

> 图 3：我们具体于特定任务的模型是通过给 BERT 加一个额外的输出层构成，所以仅需要从头学习最小数量的参数。其中（a）和（b）是序列级任务，（c）和（d）是标记级任务。图中 $E$ 表示嵌入的输入，$Ti$ 表示第 $i$ 个标记的上下文表示，[CLS] 是分类输出的特殊符号，[SEP] 是分离非连续标记（分离两个句子）序列的特殊符号。

### 4. 实验
在这一节，我们将展示 BERT 在 11 项自然语言处理任务中的微调结果。

#### 4.1 GLUE 数据集
通用语言理解评价 (GLUE General Language Understanding Evaluation) 基准（[Wang et al.(2018)](https://arxiv.org/abs/1804.07461v2)）是对多种自然语言理解任务的集合。大多数 GLUE 数据集已经存在多年，但 GLUE 的用途是（1）以分离的训练集、验证集和测试集的标准形式发布这些数据集；并且（2）建立一个评估服务器来缓解评估不一致和过度拟合测试集的问题。GLUE 不发布测试集的标签，用户必须将他们的预测上传到 GLUE 服务器进行评估，并对提交的数量进行限制。

GLUE 基准包括以下数据集，其描述最初在 [Wang et al.(2018)](https://arxiv.org/abs/1804.07461v2)中总结:

**MNLI**  多类型的自然语言推理（Multi-Genre Natural Language Inference）是一项大规模的、众包的蕴含分类任务（[Williams et al.， 2018](https://arxiv.org/abs/1704.05426v4)）。给定一对句子，目的是预测第二个句子相对于第一个句子是暗含的、矛盾的还是中立的关系。

**QQP**  Quora问题对（Quora Question Pairs）是一个二元分类任务，目的是确定两个问题在Quora上问的语义是否相等 （[Chen et al., 2018](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)）。

**QNLI**  问题自然语言推理（Question Natural Language Inference）是斯坦福问题回答数据集（[Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250v3)）已经转换为二进制分类任务的一个版本 [Wang et al.(2018)](https://arxiv.org/abs/1804.07461v2)。正类的例子是（问题，句子）对，句子中包含正确的答案，和负类的例子是来自同一段的（问题，句子）对，句子中不包含正确的答案。

**SST-2**  斯坦福情感语义树（Stanford Sentiment Treebank）数据集是一个二元单句分类任务，数据由电影评论中提取的句子组成，并对由人工对这些句子进行标注（[Socher et al., 2013](https://nlp.stanford.edu/sentiment/)）。

**CoLA**  语言可接受性单句二元分类任务语料库（Corpus of Linguistic Acceptability），它的目的是预测一个英语句子在语言学上是否 “可接受”（[Warstadt et al., 2018](https://nyu-mll.github.io/CoLA/)）。

**STS-B**  文本语义相似度基准（Semantic Textual Similarity Bench-mark ）是从新闻标题中和其它来源里提取的句子对的集合（[Cer et al., 2017](https://arxiv.org/abs/1708.00055v1)）。他们用从 1 到 5 的分数标注，表示这两个句子在语义上是多么相似。

**MRPC**  微软研究释义语料库（Microsoft Research Paraphrase Corpus）从在线新闻中自动提取的句子对组成，并用人工注解来说明这两个句子在语义上是否相等（[Dolan and Brockett, 2005.](https://www.researchgate.net/publication/228613673_Automatically_constructing_a_corpus_of_sentential_paraphrases)）。

**RTE**  识别文本蕴含（Recognizing Textual Entailment）是一个与 MNLI 相似的二元蕴含任务，只是 RTE 的训练数据更少 [Bentivogli et al., 2009](https://www.mendeley.com/catalogue/fifth-pascal-recognizing-textual-entailment-challenge/)。

**WNLI**  威诺格拉德自然语言推理（Winograd NLI）是一个来自（[Levesque et al., 2011)](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) ）的小型自然语言推理数据集。GLUE网页提示到这个数据集的构造存在问题，每一个被提交给 GLUE 的经过训练的系统在预测多数类时都低于 65.1 这个基线准确度。因此，出于对 OpenAI GPT 的公平考虑，我们排除了这一数据集。对于我们的 GLUE 提交，我们总是预测多数类。

#### 4.1.1 GLUE 结果
为了在 GLUE 上微调模型，我们按照本文第 3 节中描述的那样表示输入的句子或者句子对，并且使用最后一层的隐藏向量 $C \in \mathbb{R}^H$ 中的第一个输入标记（[CLS]）作为句子总的表示。如图3 （a）和（b）所示。在微调期间唯一引入的新的参数是一个分类层参数矩阵 $W \in \mathbb{R}^{K \times H}$，其中 $K$ 是要分类的数量。我们用 $C$ 和 $W$ 计算一个标准的分类损失，换句话说是 $log(softmax(CW^T))$。

我们在 GLUE 所有的任务中使用 32 的批次大小和 3 个周期。对于每个任务我们使用 $5e-5, 4e-5, 3e-5, 2e-5$ 的学习率来微调，然后在验证集中选择表现最好的学习率。此外，对于 $BERT_{LARGE}$ 我们发现它有时在小数据集上微调时不稳定（换句话说是，有时运行时会使结果更差），因此，我们进行了几次随机重启，并选择了在验证集上表现最好的模型。对于随机重启，我们使用相同的预训练检查点，但执行不同的数据打乱和分类器层初始化来微调模型。我们注意到，GLUE 发布的数据集不包括测试的标签，所以我们分别将 $BERT_{BASE}$ 和 $BERT_{LARGE}$ 向 GLUE 评估服务器提交结果。

结果如表 1 所示。在所有的任务上，$BERT_{BASE}$ 和 $BERT_{LARGE}$ 都比现有的系统更加出色 ，与先进水平相比，分别取得 4.4% 及 6.7% 的平均改善。请注意，除了 $BERT_{BASE}$ 含有注意力屏蔽（attention masking），$BERT_{BASE}$ 和 OpenAI GPT 的模型结构方面几乎是相同的。对于最大和最广泛使用的 GLUE 任务 MNLI，BERT 比当前最优模型获得了 4.7% 的绝对提升。在 GLUE 官方的排行榜上， $BERT_{LARGE}$ 获得了 80.4 的分数，与原榜首的 OpenAI GPT 相比截止本文写作时只获得了 72.8 分。

有趣的是， $BERT_{LARGE}$ 在所有任务中都显著优于 $BERT_{BASE}$，即使是在那些只有很少训练数据的任务上。BERT 模型大小的影响在本文 5.2 节有更深入的探讨。

<div align=center>
    <img src="zh-cn/img/bert/table_1.png" /> 
</div>
<!-- ![](zh-cn/img/bert/table_1.png) -->

> 表 1：GLUE 测试结果，由 GLUE 评估服务器评分。每个任务下面的数字表示训练示例的数量。“Average”列与官方 GLUE 评分略有不同，因为我们排除了有问题的 WNLI 数据集。OpenAI GPT = (L=12, H=768, A=12); BERTBASE = (L=12, H=768, A=12); BERTLARGE = (L=24, H=1024, A=16)。BERT 和 OpenAI GPT 都是单模型，单任务。所有结果可以从 https://gluebenchmark.com/leaderboard 和 https://blog.openai.com/language-unsupervised/ 获得。

#### 4.2 SQuAD v1.1
斯坦福问答数据集（SQuAD Standford Question Answering Dataset）是一个由 100k 个众包的问题/答案对组成的集合（[Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250v3)）。给出一个问题和一段来自维基百科包含这个问题答案的段落，我们的任务是预测这段答案文字的区间。例如:

+ 输入问题：
  Where do water droplets collide with ice crystals to form precipitation?
+ 输入段落
  ... Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud. ...
+ 输出答案
  within a cloud

这种区间预测任务与 GLUE 的序列分类任务有很大的区别，但是我们能够让 BERT 以一种直接的方式在 SQuAD 上运行。就像在 GLUE 中，我们将输入问题和段落表示为一个单一打包序列（packed sequence），其中问题使用 A 嵌入，段落使用 B 嵌入。在微调模型期间唯一需要学习的新参数是区间开始向量 $S \in \mathbb{R}^H$ 和区间结束向量 $E \in \mathbb{R}^H$。让 BERT 模型最后一层的隐藏向量的第 $i^{th}$ 输入标记被表示为 $T_i \in \mathbb{R}^H$。如图 3（c）可视化的表示。然后，计算单词 $i$ 作为答案区间开始的概率，它是 $T_i$ 和 $S$ 之间的点积并除以该段落所有单词的结果之后再 softmax:

$$P_i=\dfrac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}$$

同样的式子用来计算单词作为答案区间的结束的概率，并采用得分最高的区间作为预测结果。训练目标是正确的开始和结束位置的对数可能性。

我们使用 $5e-5$ 的学习率，32 的批次大小训练模型 3 个周期。在模型推断期间，因为结束位置与开始位置没有条件关系，我们增加了结束位置必须在开始位置之后的条件，但没有使用其他启发式。为了方便评估，把序列化后的标记区间对齐回原始未序列化的输入。

结果如表 2 中描述那样。SQuAD 使用一个高度严格的测试过程，其中提交者必须手动联系小组组织人员，然后在一个隐藏的测试集上运行他们的系统，所以我们只提交了最好的模型来测试。表中显示的结果是我们提交给小组的第一个也是唯一一个测试。我们注意到上面的结果在小组排行榜上没有最新的公共模型描述，并被允许在训练他们的模型时使用任何的公共数据。因此，我们在提交的模型中使用非常有限的数据增强，通过在 SQuAD 和 TriviaQA[(Joshi et al., 2017)](https://arxiv.org/abs/1705.03551v2) 联合训练。

我们表现最好的模型在集成模型排名中上比排名第一模型高出 1.5 个 F1 值，在一个单模型排行榜中比排名第一的模型高出 1.7（译者注：原文是 1.3） 个 F1 值。实际上，我们的单模型 BERT 就比最优的集成模型表现更优。即使只在 SQuAD 数据集上（不用 TriviaQA 数据集）我们只损失 0.1-0.4 个 F1 值，而且我们的模型输出结果仍然比现有模型的表现好很多。

<div align=center>
    <img src="zh-cn/img/bert/table_2.png" /> 
</div>
<!-- ![](zh-cn/img/bert/table_2.png) -->

> 表 2：SQuAD 结果。Ensemble BERT 是使用不同的预训练模型检查点和微调种子的 7x 模型。

#### 4.3 命名实体识别
为了评估标记任务的性能，我们在 CoNLL 2003 命名实体识别数据集（NER Named Entity Recognition）上微调 BERT 模型。该数据集由 200k 个训练单词组成，这些训练词被标注为人员、组织、地点、杂项或其他（无命名实体）。

为了微调，我们将最后一层每个单词的隐藏表示 $T_i \in \mathbb{R}^H$ 送入一个在 NER 标签集合的分类层。每个单词的分类不以周围预测为条件（换句话说，没有自回归和没有 CRF）。为了与词块（WordPiece）序列化相适应，我们把 CoNLI-序列化的（CoNLL-tokenized）的输入词输入我们的 WordPiece 序列化器，然后使用这些隐藏状态相对应的第一个块而不用预测标记为 X的块。例如：

<div align=center>
    <img src="zh-cn/img/bert/4_3_1.png" /> 
</div>
<!-- ![](zh-cn/img/bert/4_3_1.png) -->

由于单词块序列化边界是输入中已知的一部分，因此对训练和测试都要这样做。
结果如表 3 所示。$BERT_{LARGE}$ 优于现存的最优模型，使用多任务学习的交叉视野训练 [(Clark et al., 2018)](https://arxiv.org/abs/1809.08370v1)，CoNLL-2003 命名实体识别测试集上高 0.2 F1 值。

<div align=center>
    <img src="zh-cn/img/bert/table_3.png" /> 
</div>
<!-- ![](zh-cn/img/bert/table_3.png) -->

> 表 3：CoNLL-2003 命名实体识别。模型超参数使用验证集进行选择，报告的验证集和测试分数使用这些超参数进行随机五次以上的实验然后取实验的平均结果。

#### 4.4 SWAG

Adversarial Generations（SWAG）数据集由 113k 个句子对组合而成，用于评估基于常识的推理 [(Zellers et al., 2018)](https://arxiv.org/abs/1808.05326v1)。

给出一个来自视频字幕数据集的句子，任务是在四个选项中选择最合理的延续。例如:

<div align=center>
    <img src="zh-cn/img/bert/4_4_1.png" /> 
</div>
<!-- ![](zh-cn/img/bert/4_4_1.png) -->

为 SWAG 数据集调整 BERT 模型的方式与为 GLUE 数据集调整的方式相似。对于每个例子，我们构造四个输入序列，每一个都连接给定的句子（句子A）和一个可能的延续（句子B）。唯一的特定于任务的参数是我们引入向量 $V \in \mathbb{R}^{H}$，然后它点乘最后层的句子总表示 $C_i \in \mathbb{R}^H$ 为每一个选择 $i$ 产生一个分数。概率分布为 softmax 这四个选择:

$$P_i=\dfrac{e^{V \cdot C_i}}{\sum_j^4 e^{S \cdot C_j}}$$

我们使用 $2e-5$ 的学习率，16 的批次大小训练模型 3 个周期。结果如表 4 所示。$BERT_{LARGE}$ 优于作者的 ESIM+ELMo 的基线标准模型的 27.1% 。

<div align=center>
    <img src="zh-cn/img/bert/table_4.png" /> 
</div>
<!-- ![](zh-cn/img/bert/table_4.png) -->

> 表 4：SWAG 验证集和测试集准确率。测试结果由 SWAG 作者对隐藏的标签进行评分。人类的表现是用 100 个样本来衡量的，正如 SWAG 论文中描述的那样。

### 5. 消融研究（Ablation Studies）
虽然我们已经证明了非常强有力的实证结果，但到目前为止提出的结果并没有提现出 BERT 框架的每个部分具体的贡献。在本节中，我们对 BERT 的许多方面进行了消融实验，以便更好地理解每个部分的相对重要性。

#### 5.1 预训练任务的影响
我们的核心观点之一是，与之前的工作相比，BERT 的深层双向性（通过遮蔽语言模型预训练）是最重要的改进。为了证明这一观点，我们评估了两个新模型，它们使用与 $BERT_{BASE}$ 完全相同的预训练数据、微调方案和 Transformer 超参数：
1. No NSP：模型使用“遮蔽语言模型”（MLM）但是没有“预测下一句任务”（NSP）。
2. LTR & No NSP：模型使用一个从左到右（LTR）的语言模型，而不是遮蔽语言模型。在这种情况下，我们预测每个输入词，不应用任何遮蔽。在微调中也应用了仅限左的约束，因为我们发现使用仅限左的上下文进行预训练和使用双向上下文进行微调总是比较糟糕。此外，该模型未经预测下一句任务的预训练。这与OpenAI GPT有直接的可比性，但是使用更大的训练数据集、输入表示和微调方案。

结果如表 5 所示。我们首先分析了 NSP 任务所带来的影响。我们可以看到去除 NSP 对 QNLI、MNLI 和 SQuAD 的表现造成了显著的伤害。这些结果表明，我们的预训练方法对于获得先前提出的强有力的实证结果是至关重要的。

接着我们通过对比 “No NSP” 与 “LTR & No NSP” 来评估训练双向表示的影响。LTR 模型在所有任务上的表现都比 MLM 模型差，在 MRPC 和 SQuAD 上的下降特别大。对于SQuAD来说，很明显 LTR 模型在区间和标记预测方面表现很差，因为标记级别的隐藏状态没有右侧上下文。因为 MRPC 不清楚性能差是由于小的数据大小还是任务的性质，但是我们发现这种性能差是在一个完全超参数扫描和许多次随机重启之间保持一致的。

为了增强 LTR 系统，我们尝试在其上添加一个随机初始化的 BiLSTM 来进行微调。这确实大大提高了 SQuAD 的成绩，但是结果仍然比预训练的双向模型表现差得多。它还会损害所有四个 GLUE 任务的性能。

我们注意到，也可以培训单独的 LTR 和 RTL 模型，并将每个标记表示为两个模型表示的连接，就像 ELMo 所做的那样。但是：（a）这是单个双向模型参数的两倍大小；（b）这对于像 QA 这样的任务来说是不直观的，因为 RTL 模型无法以问题为条件确定答案；（c）这比深层双向模型的功能要弱得多，因为深层双向模型可以选择使用左上下文或右上下文。

<div align=center>
    <img src="zh-cn/img/bert/table_5.png" /> 
</div>
<!-- ![](zh-cn/img/bert/table_5.png) -->

> 表 5：在预训练任务中使用 $BERT_{BASE}$ 模型进行消融实验。“No NSP”表示不进行下一句预测任务来训练模型。“LTR & No NSP”表示就像 OpenAI GPT 一样，使用从左到右的语言模型不进行下一句预测任务来训练模型。“+ BiLSTM”表示在“LTR & No NSP”模型微调时添加一个随机初始化的 BiLSTM 层。

#### 5.2 模型大小的影响
在本节中，我们将探讨模型大小对微调任务准确度的影响。我们用不同的层数、隐藏单位和注意力头个数训练了许多 BERT 模型，同时使用了与前面描述的相同的超参数和训练过程。

选定 GLUE 任务的结果如表 6 所示。在这个表中，我们报告了 5 次在验证集上的微调的随机重启的平均模型准确度。我们可以看到，更大的模型在所选 4 个数据集上都带来了明显的准确率上升，甚至对于只有 3600 个训练数据的 MRPC 来说也是如此，并且与预训练任务有很大的不同。也许令人惊讶的是，相对于现有文献，我们能够在现有的模型基础上实现如此显著的改进。例如，[Vaswani et al.(2017)](https://arxiv.org/abs/1706.03762v5) 研究的最大 Transformer 为(L=6, H=1024, A=16)，编码器参数为 100M，我们所知的文献中的最大 Transformer 为(L=64, H=512, A=2)，参数为235M（[Al-Rfou et al., 2018](https://arxiv.org/abs/1808.04444v1)）。相比之下，$BERT_{BASE}$ 含有 110M 参数而 $BERT_{LARGE}$ 含有 340M 参数。

多年来人们都知道，增加模型的大小将持续提升在大型任务(如机器转换和语言建模)上的的表现，表 6 所示的由留存训练数据（held-out traing data）计算的语言模型的困惑度（perplexity）。然而，我们相信，这是第一次证明，如果模型得到了足够的预训练，那么将模型扩展到极端的规模也可以在非常小的任务中带来巨大的改进。

<div align=center>
    <img src="zh-cn/img/bert/table_6.png" /> 
</div>
<!-- ![](zh-cn/img/bert/table_6.png) -->

> 表 6：调整 BERT 的模型大小。#L = 层数；#H = 隐藏维度大小；#A = 注意力头的个数。“LM (ppl)”表示遮蔽语言模型在预留训练数据上的困惑度。


#### 5.3 训练步数的影响
图 4 显示了经过 K 步预训练模型的检查点再模型微调之后在 MNLI 验证集上的准确率。这让我们能够回答下列问题:
1. 问:BERT真的需要这么多的预训练 (128,000 words/batch * 1,000,000 steps) 来实现高的微调精度吗?
答：是的，$BERT_{BASE}$ 在 MNLI 上进行 1M 步预训练时的准确率比 500k 步提高了近 1.0%。
2. 问:遮蔽语言模型的预训练是否比 LTR 模型预训练训收敛得慢，因为每批只预测 15% 的单词，而不是每个单词?
答：遮蔽语言模型的收敛速度确实比 LTR 模型稍慢。然而，在绝对准确性方面，遮蔽语言模型几乎在训练一开始就超越 LTR 模型。

<div align=center>
    <img src="zh-cn/img/bert/figure_4.png" /> 
</div>
<!-- ![](zh-cn/img/bert/figure_4.png) -->

> 图 4：调整模型的训练步数。图中展示了已经预训练了 k 步后的模型参数在 MNLI 数据集上的再经过微调后的准确率。x 轴的值就是 k。

#### 5.4 使用 BERT 基于特征的方法
到目前为止，所有的 BERT 结果都使用了微调方法，将一个简单的分类层添加到预训练的模型中，并在一个下行任务中对所有参数进行联合微调。然而，基于特征的方法，即从预训练模型中提取固定的特征，具有一定的优势。首先，并不是所有 NLP 任务都可以通过 Transformer 编码器体系结构轻松地表示，因此需要添加特定于任务的模型体系结构。其次，能够一次性耗费大量计算预先计算训练数据的表示，然后在这种表示的基础上使用更节省计算的模型进行许多实验，这有很大的计算优势。

在本节中，我们通过在 CoNLL-2003 命名实体识别任务上生成类似于 elmo 的预训练的上下文表示来评估基于特征的方法中的 BERT 表现有多好。为此，我们使用与第 4.3 节相同的输入表示，但是使用来自一个或多个层的激活输出，而不需要对BERT的任何参数进行微调。在分类层之前，这些上下文嵌入被用作对一个初始化的两层 768 维 Bi-LSTM 的输入。

结果如表 7 所示。最佳的执行方法是从预训练的转换器的前 4 个隐藏层串联符号表示，这只比整个模型的微调落后 0.3 F1 值。这说明 BERT 对于微调和基于特征的方法都是有效的。

<div align=center>
    <img src="zh-cn/img/bert/table_7.png" /> 
</div>
<!-- ![](zh-cn/img/bert/table_7.png) -->

> 表 7：在 CoNLL-2003 命名实体识别上使用基于特征的方法，并调整 BERT 层数。来自指定层的激活输出被组合起来，并被送到一个两层的 BiLSTM 中，而不需要反向传播到 BERT。


### 6. 结论
最近，由于使用语言模型进行迁移学习而取得的实验提升表明，丰富的、无监督的预训练是许多语言理解系统不可或缺的组成部分。特别是，这些结果使得即使是低资源（少量标签的数据集）的任务也能从非常深的单向结构模型中受益。我们的主要贡献是将这些发现进一步推广到深层的双向结构，使同样的预训练模型能够成功地广泛地处理 NLP 任务。

虽然这些实证结果很有说服力，在某些情况下甚至超过了人类的表现，但未来重要的工作是研究 BERT 可能捕捉到的或不捕捉到的语言现象。

### 参考文献
所有参考文献按论文各小节中引用顺序排列，多次引用会多次出现在下面的列表中。

**Abstract 摘要中的参考文献**

|BERT 文中简写|原始标论文标题|其它|
|-|-|-|
|Peters et al., 2018|[Deep contextualized word representations](https://arxiv.org/abs/1802.05365v2)|ELMo|
|Radford et al., 2018|[Improving Language Understanding with Unsupervised Learning](https://blog.openai.com/language-unsupervised/)|OpenAI GPT|


**1. Introduction 介绍中的参考文献**

|BERT 文中简写|原始标论文标题|其它|
|-|-|-|
|Peters et al., 2018|[Deep contextualized word representations](https://arxiv.org/abs/1802.05365v2)|ELMo|
|Radford et al., 2018|[Improving Language Understanding with Unsupervised Learning](https://blog.openai.com/language-unsupervised/)|OpenAI GPT|
|Dai and Le, 2015|[Semi-supervised sequence learning. In Advances in neural information processing systems, pages 3079–3087](http://papers.nips.cc/paper/5949-semi-supervised-sequence-learning)|AndrewMDai and Quoc V Le. 2015|
|Howard and Ruder, 2018|[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146v5)|**ULMFiT**；Jeremy Howard and Sebastian Ruder.|
|Bow-man et al., 2015|[A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326v1)|Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning.|
|Williams et al., 2018|[A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426v4)|Adina Williams, Nikita Nangia, and Samuel R Bowman.|
|Dolan and Brockett, 2005|[Automatically constructing a corpus of sentential paraphrases](https://www.researchgate.net/publication/228613673_Automatically_constructing_a_corpus_of_sentential_paraphrases)|William B Dolan and Chris Brockett. 2005.|
|Tjong Kim Sang and De Meulder, 2003|[Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](http://www.oalib.com/paper/4018980)|Erik F Tjong Kim Sang and Fien De Meulder. 2003.|
|Rajpurkar et al., 2016|[SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250v3)|SQuAD|
|Taylor, 1953|["Cloze Procedure": A New Tool For Measuring Readability](https://www.researchgate.net/publication/232539913_Cloze_Procedure_A_New_Tool_For_Measuring_Readability)|Wilson L Taylor. 1953.|

**2. Related Work 相关工作中的参考文献**

|BERT 文中简写|原始标论文标题|其它|
|-|-|-|
|Brown et al., 1992|[Class-based n-gram models of natural language](https://dl.acm.org/citation.cfm?id=176316)|Peter F Brown, Peter V Desouza, Robert L Mercer, Vincent J Della Pietra, and Jenifer C Lai. 1992.|
|Ando and Zhang, 2005|[A Framework for Learning Predictive Structures from Multiple Tasks and Unlabeled Data](http://academictorrents.com/details/f4470eb8bc3a6f697df61bde319fd56e3a9d6733)|Rie Kubota Ando and Tong Zhang. 2005.|
|Blitzer et al., 2006|[Domain adaptation with structural correspondence learning](https://dl.acm.org/citation.cfm?id=1610094)|John Blitzer, Ryan McDonald, and Fernando Pereira.2006.|
|Collobert and Weston, 2008|[A Unified Architecture for Natural Language Processing](https://www.researchgate.net/publication/200044432_A_Unified_Architecture_for_Natural_Language_Processing)|Ronan Collobert and Jason Weston. 2008.|
|Mikolov et al., 2013|[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546v1)|CBOW Model；Skip-gram Model|
|Pennington et al., 2014|[GloVe: Global Vectors for Word Representation](http://www.aclweb.org/anthology/D14-1162)|GloVe|
|Turian et al., 2010|[Word Representations: A Simple and General Method for Semi-Supervised Learning](https://www.researchgate.net/publication/220873681_Word_Representations_A_Simple_and_General_Method_for_Semi-Supervised_Learning)|Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010.|
|Kiros et al., 2015|[Skip-Thought Vectors](https://arxiv.org/abs/1506.06726v1)|Skip-Thought Vectors|
|Logeswaran and Lee, 2018|[An efficient framework for learning sentence representations](https://arxiv.org/abs/1803.02893v1)|Lajanugen Logeswaran and Honglak Lee. 2018.|
|Le and Mikolov, 2014|[Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053v2)|Quoc Le and Tomas Mikolov. 2014.|
|Peters et al., 2017|[Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/abs/1705.00108v1)|Matthew Peters, Waleed Ammar, Chandra Bhagavatula, and Russell Power. 2017.|
|Peters et al., 2018|[Deep contextualized word representations](https://arxiv.org/abs/1802.05365v2)|ELMo|
|Rajpurkar et al., 2016|[SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250v3)|SQuAD|
|Socher et al., 2013|[Deeply Moving: Deep Learning for Sentiment Analysis](https://nlp.stanford.edu/sentiment/)|SST-2|
|Tjong Kim Sang and De Meulder, 2003|[Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](http://www.oalib.com/paper/4018980)|Erik F Tjong Kim Sang and Fien De Meulder. 2003.|
|Dai and Le, 2015|[Semi-supervised sequence learning. In Advances in neural information processing systems, pages 3079–3087](http://papers.nips.cc/paper/5949-semi-supervised-sequence-learning)|AndrewMDai and Quoc V Le. 2015|
|Howard and Ruder, 2018|[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146v5)|**ULMFiT**；Jeremy Howard and Sebastian Ruder.|
|Radford et al., 2018|[Improving Language Understanding with Unsupervised Learning](https://blog.openai.com/language-unsupervised/)|OpenAI GPT|
|Wang et al.(2018)|[GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461v2)|GLUE|
|Con-neau et al., 2017|[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://www.aclweb.org/anthology/D17-1070)|Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes. 2017.|
|McCann et al., 2017|[Learned in Translation: Contextualized Word Vectors](https://einstein.ai/static/images/pages/research/cove/McCann2017LearnedIT.pdf)|Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017.|
|Deng et al.|[ImageNet: A large-scale hierarchical image database](https://ieeexplore.ieee.org/document/5206848)|J. Deng,W. Dong, R. Socher, L.-J. Li, K. Li, and L. FeiFei. 2009.|
|Yosinski et al., 2014|[How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792v1)|Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. 2014.|

**3. BERT 中的参考文献**

|BERT 文中简写|原始标论文标题|其它|
|-|-|-|
|Vaswani et al. (2017)|[Attention Is All You Need](https://arxiv.org/abs/1706.03762v5)|Transformer|
|Wu et al., 2016|[Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144v2)|WordPiece|
|Taylor, 1953|["Cloze Procedure": A New Tool For Measuring Readability](https://www.researchgate.net/publication/232539913_Cloze_Procedure_A_New_Tool_For_Measuring_Readability)|Wilson L Taylor. 1953.|
|Vincent et al., 2008|[Extracting and composing robust features with denoising autoencoders](https://www.researchgate.net/publication/221346269_Extracting_and_composing_robust_features_with_denoising_autoencoders)|denoising auto-encoders|
|Zhu et al., 2015|[Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](https://arxiv.org/abs/1506.06724v1)|BooksCorpus (800M words)|
|Chelba et al., 2013|[One Billion Word Benchmark for Measuring Progress in Statistical Language Modeling](https://arxiv.org/abs/1312.3005v3)|Billion Word Benchmark corpus|
|Hendrycks and Gimpel, 2016|[Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415v3)|GELUs|

**4. Experiments 实验中的参考文献**

|BERT 文中简写|原始标论文标题|其它|
|-|-|-|
|Wang et al.(2018)|[GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461v2)|GLUE|
|Williams et al., 2018|[A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426v4)|MNLI|
|Chen et al., 2018|[First Quora Dataset Release: Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)|QQP|
|Rajpurkar et al., 2016|[SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250v3)|QNLI|
|Socher et al., 2013|[Deeply Moving: Deep Learning for Sentiment Analysis](https://nlp.stanford.edu/sentiment/)|SST-2|
|Warstadt et al., 2018|[The Corpus of Linguistic Acceptability](https://nyu-mll.github.io/CoLA/)|CoLA|
|Cer et al., 2017|[SemEval-2017 Task 1: Semantic Textual Similarity - Multilingual and Cross-lingual Focused Evaluation](https://arxiv.org/abs/1708.00055v1)|STS-B|
|Dolan and Brockett, 2005|[Automatically constructing a corpus of sentential paraphrases](https://www.researchgate.net/publication/228613673_Automatically_constructing_a_corpus_of_sentential_paraphrases)|MRPC|
|Bentivogli et al., 2009|[The fifth pascal recognizing textual entailment challenge](https://www.mendeley.com/catalogue/fifth-pascal-recognizing-textual-entailment-challenge/)|RTE|
|Levesque et al., 2011|[The winograd schema challenge. In Aaai spring symposium: Logical formalizations of commonsense reasoning, volume 46, page 47.](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html)|WNLI|
|Rajpurkar et al., 2016|[SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250v3)|SQuAD|
|Joshi et al., 2017|[TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://arxiv.org/abs/1705.03551v2)|TriviaQA|
|Clark et al., 2018|[Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/abs/1809.08370v1)||
|Zellers et al., 2018|[SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference](https://arxiv.org/abs/1808.05326v1)|SWAG|

**5. Ablation Studies 消融研究中的参考文献**

|BERT 文中简写|原始标论文标题|其它|
|-|-|-|
|Vaswani et al. (2017)|[Attention Is All You Need](https://arxiv.org/abs/1706.03762v5)|Transformer|
|Al-Rfou et al., 2018|[Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/abs/1808.04444v1)||


### Example: BERT实现情感极性分析

我们之前重新训练了一个小的BERT的模型，将该模型应用于情感极性分析的分类任务的fine-tune,使得在情感极性分析任务上在训练集和测试集上的AUC均达到95%以上。该项目可以通过该ropo <https://github.com/DataXujing/bert-chinese-classification> 获得！


------


## ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

<!-- https://wmathor.com/index.php/archives/1480/ -->
<!-- https://blog.csdn.net/jiaowoshouzi/article/details/102320781 -->
<!-- https://blog.csdn.net/u012526436/article/details/101924049 -->
<!-- https://blog.csdn.net/weixin_37947156/article/details/101529943 -->

<!-- https://www.bilibili.com/video/BV1C7411c7Ag?from=search&seid=11518776277468921446 -->
<!-- https://www.bilibili.com/video/BV1uA411n7sC?from=search&seid=11518776277468921446 -->

RoBERTa没霸榜几天，这不Google就又放大招，这次的新模型不再是简单的的升级，而是采用了全新的参数共享机制，反观其他升级版BERT模型，基本都是添加了更多的预训练任务，增大数据量等轻微的改动。这次ALBERT的改进，不仅提升了模型的整体效果再一次拿下来各项榜单的榜首，而且参数量相比BERT来说少了很多。

对于预训练模型来说，提升模型的大小是能对下游任务的效果有一定提升，然而如果进一步提升模型规模，势必会导致显存或者内存出现OOM的问题，长时间的训练也可能导致模型出现退化的情况。为了解决这些问题，Google提出了ALBERT，该模型提出了两种减少内存的方法，同时提升了训练速度(很遗憾对于较深较宽的模型推断时间变慢了），其次改进了BERT中的NSP的预训练任务。

**内存限制**

考虑一个包含一个输入节点，两个隐藏节点和一个输出节点的简单神经网络。即使是这样一个简单的神经网络，由于每个节点有权重和偏差，因此总共有 7 个参数需要学习

<div align=center>
    <img src="zh-cn/img/bert/albert/p1.png" /> 
</div>

BERT-large 是一个复杂的模型，它有 24 个隐藏层，在前馈网络和多头注意力机制中有很多节点，总共有 3.4 亿个参数，如果想要从零开始训练，需要花费大量的计算资源

<div align=center>
    <img src="zh-cn/img/bert/albert/p2.png" /> 
</div>

**模型退化**

最近在 NLP 领域的研究趋势是使用越来越大的模型，以获得更好的性能。ALBERT 的研究表明，无脑堆叠模型参数可能导致效果降低。在论文中，作者做了一个有趣的实验

>如果更大的模型可以带来更好的性能，为什么不将最大的 BERT 模型 (BERT-large) 的隐含层单元增加一倍，从 1024 个单元增加到 2048 个单元呢？

他们称之为 "BERT-xlarge"。令人惊讶的是，无论是在语言建模任务还是阅读理解测试（RACE）中，这个更大的模型的表现都不如 BERT-large,从原文给出的图中（下图），我们可以看到性能是如何下降的

<div align=center>
    <img src="zh-cn/img/bert/albert/p3.png" /> 
</div>

ALBERT也是采用和BERT一样的Transformer的encoder结果，激活函数使用的也是GELU，在讲解下面的内容前，我们规定几个参数，词的embedding我们设置为E，encoder的层数我们设置为L，hidden size即encoder的输出值的维度我们设置为H，前馈神经网络的节点数设置为4H，attention的head个数设置为H/64。

在ALBERT中主要有三个改进方向。

### 1.对Embedding因式分解（Factorized embedding parameterization）

原始的 BERT 模型以及各种依据 Transformer 的预训连语言模型都有一个共同特点，即 `E=H`，其中 `E` 指的是 Embedding Dimension，`H` 指的是 Hidden Dimension。这就会导致一个问题，当提升 Hidden Dimension 时，Embedding Dimension 也需要提升，最终会导致参数量呈平方级的增加。所以 ALBERT 的作者将 E 和 H 进行解绑，具体的操作就是在 Embedding 后面加入一个矩阵进行维度变换。E 的维度是不变的，如果 H 增大了，我们只需要在 E 后面进行一个升维操作即可

<div align=center>
    <img src="zh-cn/img/bert/albert/p4.png" /> 
</div>


所以，ALBERT 不直接将原本的 one-hot 向量映射到 hidden space size of H，而是分解成两个矩阵，原本参数数量为 `V∗H`，V 表示的是 Vocab Size。分解成两步则减少为 `V∗E+E∗H`，当 H 的值很大时，这样的做法能够大幅降低参数数量

> V∗H=30000∗768=23,040,000
> V∗E+E∗H=30000∗256+256∗768=7,876,608
> 举个例子，当 V 为 30000，H 为 768，E 为 256 时，参数量从 2300 万降低到 780 万


在BERT、XLNet、RoBERTa中，词表的embedding size(E)和transformer层的hidden size(H)都是相等的，这个选择有两方面缺点：

+ 从建模角度来讲，wordpiece向量应该是不依赖于当前内容的(context-independent)，而transformer所学习到的表示应该是依赖内容的。所以把E和H分开可以更高效地利用参数，因为理论上存储了context信息的H要远大于E。
+ 从实践角度来讲，NLP任务中的vocab size本来就很大，如果`E=H`的话，模型参数量就容易很大，而且embedding在实际的训练中更新地也比较稀疏。

因此作者使用了小一些的`E(64、128、256、768)`，训练一个独立于上下文的embedding`(VxE)`，之后计算时再投影到隐层的空间(乘上一个`ExH`的矩阵)，相当于做了一个因式分解。

下图是E选择不同值的一个实验结果，尴尬的是，在不采用参数共享优化方案时E设置为768效果反而好一些，在采用了参数共享优化方案时E取128效果更好一些。

<div align=center>
    <img src="zh-cn/img/bert/albert/p5.png" /> 
</div>


### 2.跨层的参数共享（Cross-layer parameter sharing）

<div align=center>
    <img src="zh-cn/img/bert/albert/p6.png" /> 
</div>

在ALBERT还提出了一种参数共享的方法，Transformer中共享参数有多种方案，只共享全连接层，只共享attention层，ALBERT结合了上述两种方案，全连接层与attention层都进行参数共享，也就是说共享encoder内的所有参数，同样量级下的Transformer采用该方案后实际上效果是有下降的，但是参数量减少了很多，训练速度也提升了很多。

下图是BERT与ALBERT的一个对比，以base为例，BERT的参数是108M，而ALBERT仅有12M，但是效果的确相比BERT降低了两个点。由于其速度快的原因，我们再以BERT-xlarge为参照标准其参数是1280M，假设其训练速度是1，ALBERT的xxlarge版本的训练速度是其1.2倍，并且参数也才223M，评判标准的平均值也达到了最高的88.7

<div align=center>
    <img src="zh-cn/img/bert/albert/p7.png" /> 
</div>

除了上述说了训练速度快之外，ALBERT每一层的输出的embedding相比于BERT来说震荡幅度更小一些。下图是不同的层的输出值的L2距离与cosine相似度，可见参数共享其实是有稳定网络参数的作用的。

<div align=center>
    <img src="zh-cn/img/bert/albert/p8.png" /> 
</div>



### 3.句间连贯（Inter-sentence coherence loss）（SOP）

后BERT时代很多研究(XLNet、RoBERTa)都发现next sentence prediction没什么用处，所以作者也审视了一下这个问题，认为NSP之所以没用是因为这个任务不仅包含了句间关系预测，也包含了主题预测，而主题预测显然更简单些（比如一句话来自新闻财经，一句话来自文学小说），模型会倾向于通过主题的关联去预测。因此换成了SOP(sentence order prediction)，预测两句话有没有被交换过顺序。实验显示新增的任务有1个点的提升。

BERT的NSP任务实际上是一个二分类，训练数据的正样本是通过采样同一个文档中的两个连续的句子，而负样本是通过采用两个不同的文档的句子。该任务主要是希望能提高下游任务的效果，例如NLI自然语言推理任务。但是后续的研究发现该任务效果并不好，主要原因是因为其任务过于简单。NSP其实包含了两个子任务，主题预测与关系一致性预测，但是主题预测相比于关系一致性预测简单太多了，并且在MLM任务中其实也有类型的效果。

这里提一下为啥包含了主题预测，因为正样本是在同一个文档中选取的，负样本是在不同的文档选取的，假如我们有2个文档，一个是娱乐相关的，一个是新中国成立70周年相关的，那么负样本选择的内容就是不同的主题，而正样都在娱乐文档中选择的话预测出来的主题就是娱乐，在新中国成立70周年的文档中选择的话就是后者这个主题了。

在ALBERT中，为了只保留一致性任务去除主题识别的影响，提出了一个新的任务 sentence-order prediction（SOP），SOP的正样本和NSP的获取方式是一样的，负样本把正样本的顺序反转即可。SOP因为实在同一个文档中选的，其只关注句子的顺序并没有主题方面的影响。并且SOP能解决NSP的任务，但是NSP并不能解决SOP的任务，该任务的添加给最终的结果提升了一个点。

<div align=center>
    <img src="zh-cn/img/bert/albert/p9.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/bert/albert/p10.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/bert/albert/p11.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/bert/albert/p12.png" /> 
</div>


### 4.其他Trick


**1.删除Dropout**

Dropout个人觉得可以一开始就不用，而不是训练一段时间再关掉。学都学不动，防啥过拟合啊。模型的内部任务（MLM，SOP等等）都没有过拟合dropout是为了降低过拟合而增加的机制，所以对于bert而言是弊大于利的机制

<div align=center>
    <img src="zh-cn/img/bert/albert/p13.png" /> 
</div>

如上图所示，ALBERT的最大模型在训练1M步后仍然没有过拟合，于是作者决定删除dropout，进一步提高模型能力。

**2.LAMB优化器**

为加快训练速度，使用LAMB做为优化器。LAMB优化器使得我们可以训练，特别大的批次batch_size，如高达6万。

+ https://arxiv.org/abs/1904.00962

**3.Segments-Pair**

BERT为了加速训练，前90%的steps使用了128个token的短句子，最后10%才使用512个token的长句子训练位置向量。
ALBERT貌似90%的情况下使用512的segment，从数据上看，更长的数据提供更多的上下文信息，可能显著提升模型的能力。

**4.Masked-ngram-LM**

BERT的MLM目标是随机MASK 15%的词来预测，ALBERT预测的是N-gram片段，包含更多的语义信息，每个片段长度`n`（最大为3），根据概率公式计算得到。比如uni-gram、bi-gram、tri-gram的的概率分别为`6/11`、`3/11`、`2/11`。越长概率越小：

<div align=center>
    <img src="zh-cn/img/bert/albert/p14.png" /> 
</div>

<!-- 本项目中目前使用的是在中文上做whole word mask，稍后会更新一下与n-gram mask的效果对比。n-gram从spanBERT中来。 -->


### 5.总结

刚开始看这篇文章是很惊喜的，因为它直接把同等量级的 BERT 缩小了 10 + 倍，让普通用户有了运行可能。但是仔细看了实验后才发现参数量的减小是需要付出代价的

<div align=center>
    <img src="zh-cn/img/bert/albert/p15.png" /> 
</div>

需要注意的是，Speedup 是训练时间而不是 Inference 时间。Inference 时间并未得到改善，因为即使是使用了共享参数机制，还是得跑完 12 层 Encoder，故 Inference 时间跟 BERT 是差不多的。

实验用的参数如下

<div align=center>
    <img src="zh-cn/img/bert/albert/p16.png" /> 
</div>

可以得出的结论是：

+ 在相同的训练时间下，ALBERT 得到的效果确实比 BERT 好
+ 在相同的 Inference 时间下，ALBERT base 和 large 的效果都没有 BERT 好，而且差了 2-3 个点，作者在最后也提到了会继续寻找提高速度的方法（Sparse attention 和 Block attention）

另外，结合 Universal Transformer 可以想到的是，在训练和 Inference 阶段可以动态地调整 Transformer 层数（告别 12、24、48 的配置）。同时可以想办法去避免纯参数共享带来的效果下降，毕竟 Transformer 中越深层学到的任务相关信息越多，可以改进 Transformer 模块，加入记忆单元、每层个性化的 Embedding。



## RoBERTa: A Robustly Optimized BERT Pretraining Approach

<!-- https://www.cnblogs.com/ffjsls/p/12260785.html -->
<!-- https://mp.weixin.qq.com/s/iqY94mynw_as5l_L7xn7-w -->

先来回顾一下Bert中的一些细节：

+ 在输入上，Bert的输入是两个segment，其中每个segment可以包含多个句子，两个segment用`[SEP]`拼接起来。
+ 模型结构上，使用Transformer，这点跟Roberta是一致的。
+学习目标上，使用两个目标：
    - Masked Language Model(MLM): 其中`15%`的token要被Mask，在这`15%`里，有`80%`被替换成`[Mask]`标记，有`10%`被随机替换成其他token，有`10%`保持不变。
    - Next Sentence Prediction: 判断segment对中第二个是不是第一个的后续。随机采样出`50%`是和`50%`不是。
+ Optimizations:
    - Adam, beta1=0.9, beta2=0.999, epsilon=1e-6, L2 weight decay=0.01
    - learning rate, 前10000步会增长到1e-4, 之后再线性下降。
    - dropout=0.1
    - GELU激活函数
    - 训练步数：1M
    - mini-batch: 256
    - 输入长度: 512
+ Data
    - BookCorpus + English Wiki = 16GB


<!-- Roberta在如下几个方面对Bert进行了调优：

- Masking策略——静态与动态
- 模型输入格式与Next Sentence Prediction
- Large-Batch
- 输入编码
- 大语料与更长的训练步数
 -->

RoBERTa是在论文《RoBERTa: A Robustly Optimized BERT Pretraining Approach》中被提出的。此方法属于BERT的强化版本，也是BERT模型更为精细的调优版本。RoBERTa主要在三方面对之前提出的BERT做了该进，其一是模型的具体细节层面，改进了优化函数；其二是训练策略层面，改用了动态掩码的方式训练模型，证明了NSP（Next Sentence Prediction）训练策略的不足，采用了更大的batch size；其三是数据层面，一方面使用了更大的数据集，另一方面是使用BPE（Byte-Pair Encoding ）来处理文本数据。

### 1.RoBERTa对一般BERT的模型细节进行了优化

**Optimization**

​原始BERT优化函数采用的是Adam默认的参数，其中`β1=0.9,β2=0.999`，在RoBERTa模型中考虑采用了更大的batches，所以将`β2`改为了`0.98`。

### 2.RoBARTa对一般BERT的训练策略进行了优化

（1）动态掩码与静态掩码

**原始静态mask：**
BERT中是准备训练数据时，每个样本只会进行一次随机mask（因此每个epoch都是重复），后续的每个训练步都采用相同的mask，这是原始静态mask，即单个静态mask，这是原始 BERT 的做法。

**修改版静态mask：**
在预处理的时候将数据集拷贝 10 次，每次拷贝采用不同的 mask（总共40 epochs，所以每一个mask对应的数据被训练4个epoch）。这等价于原始的数据集采用10种静态 mask 来训练 40个 epoch。

**动态mask：**
并没有在预处理的时候执行 mask，而是在每次向模型提供输入时动态生成 mask，所以是时刻变化的。
不同模式的实验效果如下表所示。其中 reference 为BERT 用到的原始静态 mask，static 为修改版的静态mask。

<div align=center>
    <img src="zh-cn/img/bert/roberta/p1.png" /> 
</div>

（2）对NSP训练策略的探索

​ 为了探索NSP训练策略对模型结果的影响，将以下4种训练方式及进行对比：

**SEGMENT-PAIR + NSP：**
这是原始 BERT 的做法。输入包含两部分，每个部分是来自同一文档或者不同文档的 segment （segment 是连续的多个句子），这两个segment 的token总数少于 512 。预训练包含 MLM 任务和 NSP 任务。

**SENTENCE-PAIR + NSP：**
输入也是包含两部分，每个部分是来自同一个文档或者不同文档的单个句子，这两个句子的token 总数少于 512。由于这些输入明显少于512 个tokens，因此增加batch size的大小，以使 tokens 总数保持与SEGMENT-PAIR + NSP 相似。预训练包含 MLM 任务和 NSP 任务。

**FULL-SENTENCES：**
输入只有一部分（而不是两部分），来自同一个文档或者不同文档的连续多个句子，token 总数不超过 512 。输入可能跨越文档边界，如果跨文档，则在上一个文档末尾添加文档边界token 。预训练不包含 NSP 任务。

**DOC-SENTENCES：**
输入只有一部分（而不是两部分），输入的构造类似于FULL-SENTENCES，只是不需要跨越文档边界，其输入来自同一个文档的连续句子，token 总数不超过 512 。在文档末尾附近采样的输入可以短于 512个tokens， 因此在这些情况下动态增加batch size大小以达到与 FULL-SENTENCES 相同的tokens总数。预训练不包含 NSP 任务。

​ 以下是论文中4种方法的实验结果：

<div align=center>
    <img src="zh-cn/img/bert/roberta/p2.png" /> 
</div>

从实验结果来看，如果在采用NSP loss的情况下，将SEGMENT-PAIR与SENTENCE-PAIR 进行对比，结果显示前者优于后者。发现单个句子会损害下游任务的性能，可能是如此模型无法学习远程依赖。接下来把重点放在没有NSP loss的FULL-SENTENCES上，发现其在四种方法中结果最好。可能的原因：原始 BERT 实现采用仅仅是去掉NSP的损失项，但是仍然保持 SEGMENT-PARI的输入形式。最后，实验还发现将序列限制为来自单个文档(doc-sentence)的性能略好于序列来自多个文档(FULL-SENTENCES)。但是 DOC-SENTENCES 策略中，位于文档末尾的样本可能小于 512 个 token。为了保证每个 batch 的 token 总数维持在一个较高水平，需要动态调整 batch-size。出于处理方便，后面采用DOC-SENTENCES输入格式。

（3）Training with large batches

虽然在以往的经验中，当学习速率适当提高时，采用非常大mini-batches的训练既可以提高优化速度，又可以提高最终任务性能。但是论文中通过实验，证明了更大的batches可以得到更好的结果，实验结果下表所示。

<div align=center>
    <img src="zh-cn/img/bert/roberta/p3.png" /> 
</div>

​论文考虑了并行计算等因素，在后续的实验中使用`batch size=8k`进行训练。

### 3.RoBARTa在数据层面对模型进行了优化

（1）使用了更大的训练数据集

将16G的数据集提升到160G数据集，并改变多个steps，寻找最佳的超参数。

<div align=center>
    <img src="zh-cn/img/bert/roberta/p4.png" /> 
</div>

（2）Text Encoding

字节对编码(BPE)(Sennrich et al.,2016)是字符级和单词级表示的混合，该编码方案可以处理自然语言语料库中常见的大量词汇。BPE不依赖于完整的单词，而是依赖于子词(sub-word)单元，这些子词单元是通过对训练语料库进行统计分析而提取的，其词表大小通常在 1万到 10万之间。当对海量多样语料建模时，unicode characters占据了该词表的大部分。Radford et al.(2019)的工作中介绍了一个简单但高效的BPE， 该BPE使用字节对而非unicode characters作为子词单元。

总结下两种BPE实现方式：

+ 基于 char-level ：原始 BERT 的方式，它通过对输入文本进行启发式的词干化之后处理得到。
+ 基于 bytes-level：与 char-level 的区别在于bytes-level 使用 bytes 而不是 unicode 字符作为 sub-word 的基本单位，因此可以编码任何输入文本而不会引入 UNKOWN 标记。

当采用 bytes-level 的 BPE 之后，词表大小从3万（原始 BERT 的 char-level ）增加到5万。这分别为 BERT-base和 BERT-large增加了1500万和2000万额外的参数。之前有研究表明，这样的做法在有些下游任务上会导致轻微的性能下降。但是本文作者相信：这种统一编码的优势会超过性能的轻微下降。且作者在未来工作中将进一步对比不同的encoding方案。

### 总结

从上面的各种实验结果中看，可以得到如下结论：

+ NSP不是必须的loss
+ Mask的方式虽不是最优但是已接近。
+ 增大batch size和增大训练数据能带来较大的提升。

由于RoBERTa出色的性能，现在很多应用都是基于RoBERTa而不是原始的Bert去微调了。继续增大数据集，还有没有可能提升？数据集的量与所带来的提升是一个什么分布？
不管是Bert还是Roberta，训练时间都很长，如何进行优化？

**参考文献:**

[1].Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).

[2].https://mp.weixin.qq.com/s/iqY94mynw_as5l_L7xn7-w

[3].https://www.cnblogs.com/ffjsls/p/12260785.html

[4].https://github.com/pytorch/fairseq