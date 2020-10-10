## SpanBERT: Improving Pre-training by Representing and Predicting Spans

<!-- https://zhuanlan.zhihu.com/p/103203220 -->
<!-- https://blog.csdn.net/weixin_37947156/article/details/99210514 -->
<!-- https://cloud.tencent.com/developer/article/1476168 -->

<!-- https://zhuanlan.zhihu.com/p/75893972 -->


<div align=center>
    <img src="zh-cn/img/spanbert/p1.jpg" /> 
</div>


在本文中，作者提出了一个新的分词级别的预训练方法 SpanBERT ，其在现有任务中的表现优于 BERT ，并在问答、指代消解等分词选择任务中取得了较大的进展。对 BERT 模型进行了如下改进：

+ 提出了更好的 **Span Mask 方案**，SpanBERT 不再对随机的单个 token 添加掩膜，而是随机对邻接分词添加掩膜；
+ 通过加入 **Span Boundary Objective (SBO) 训练目标**，通过使用分词边界的表示来预测被添加掩膜的分词的内容，不再依赖分词内单个 token 的表示，增强了 BERT 的性能，特别在一些与 Span 相关的任务，如抽取式问答；
+ 用实验获得了和 XLNet 类似的结果，发现不加入 **Next Sentence Prediction (NSP) 任务，直接用连续一长句训练效果更好**。

整体模型结构如下:

<div align=center>
    <img src="zh-cn/img/spanbert/p2.png" /> 
</div>

接下来我会详细介绍上面三个 idea，包括相关消融实验，同时给出一些自己看法。

**Span Masking**

首先什么是 Span Masking，和一般 BERT 训练有何不同。

对于原始的 BERT，训练时，会随机选取整句中的最小输入单元 token 来进行遮盖。因为用到 Byte Pair Encoding （BPE）技术，所以也可以把这些最小单元当作是子词（subword或wordpiece），比如说superman，分成 super+man 两个子词。

但这样会让本来应该有强相关的一些连在一起的字词，在训练时是割裂开来的。因此我们就会想到，那能不能遮盖掉这样连在一起的片段训练呢？当然可以。

首先想到的做法，既然现在遮盖子词，那能不能直接遮盖整个词，比如说对于 super + man，只要遮盖就两个同时遮盖掉，这便是 Google 放出的 BERT WWM 模型所做的。

<div align=center>
    <img src="zh-cn/img/spanbert/p3.png" /> 
</div>

于是能不能进一步，因为有些实体是几个词组成的，直接将这个实体都遮盖掉。因此百度在 ERNIE 模型中，就引入命名实体（Named Entity）外部知识，遮盖掉实体单元，进行训练。

<div align=center>
    <img src="zh-cn/img/spanbert/p4.jpg" /> 
</div>

以上两种做法比原始做法都有些提升。但这两种做法会让人认为，或许必须得引入类似词边界信息才能帮助训练。但前不久的 MASS 模型，却表明可能并不需要，随机遮盖可能效果也很好，于是就有本篇的 idea：

根据**几何分布**，先随机选择一段（span）的长度，之后再根据均匀分布随机选择这一段的起始位置，最后按照长度遮盖。文中使用**几何分布**取$p=0.2$，最大长度只能是 `10`，利用此方案获得平均采样长度分布。

<div align=center>
    <img src="zh-cn/img/spanbert/p5.png" /> 
</div>

通过采样，平均被遮盖长度是**3.8**个词的长度。其实这里我有手动算了下几何分布的期望,其推导公式如下：

<div align=center>
    <img src="zh-cn/img/spanbert/p6.jpg" /> 
</div>

却发现还是少了点，这也是我最初觉得不是这样计算的主要原因，然而这里犯错的主要原因是没有对剩下的概率重新 normalize，操作完后就会发现得：

<div align=center>
    <img src="zh-cn/img/spanbert/p7.png" /> 
</div>

正好是论文中长度，特别是将概率 normalize 之后，上图中几何分布的数值也获得了解释。

这便是文中用到的 Span Masking，消融实验中，作者们用此方法和其他遮盖方法进行了对比，比如原始 BERT 的子词遮盖，或 WWM 的对整词遮盖，亦或 ERNIE 的实体遮盖，还有对名词短语遮盖。


<div align=center>
    <img src="zh-cn/img/spanbert/p8.jpg" /> 
</div>

结果发现除了 Coreference Resolution (指代消解) 任务，论文中用到的 Random Span 方法普遍更优。但作者们之后发现，只要给 Random Span 再加上 SBO 训练目标，就能大大提高这个方法的表现。

**SBO**

Span Boundary Objective 是该论文加入的新训练目标，希望被遮盖 Span 边界的词向量，能学习到 Span 的内容。或许作者想通过这个目标，让模型在一些需要 Span 的下游任务取得更好表现，结果表明也正如此。

具体做法是，在训练时取 Span 前后边界的两个词，值得指出，这两个词不在 Span 内，然后用这两个词向量加上 Span 中被遮盖掉词的位置向量，来预测原词。

+ $x_i$ : 在span中的每一个 token 表示
+ $y_i$ : 在span中的每一个 token 用来预测 [公式] 的输出
+ $x_{s-1}$ : 代表了span的开始的前一个token的表示
+ $x_{e+1}$ : 代表了span的结束的后一个token的表示
+ $p_i$: 代表了$x_i$的位置

$y_i=f(x_{s-1},x_{e+1},p_i)$ 其中 $f(.)$是一个两层的feed-foreward的神经网络 with Gelu 和layer normalization

<div align=center>
    <img src="zh-cn/img/spanbert/p9.png" /> 
</div>

详细做法是将词向量和位置向量拼接起来，过两层全连接层，很简单。

最后预测 Span 中原词时获得一个新损失，就是 SBO 目标的损失，之后将这个损失和 BERT 的 **Mased Language Model （MLM）**的损失加起来，一起用于训练模型。

<div align=center>
    <img src="zh-cn/img/spanbert/p10.png" /> 
</div>

加上 SBO 后效果普遍提高，特别是之前的指代消解任务，提升很大。


<div align=center>
    <img src="zh-cn/img/spanbert/p11.png" /> 
</div>

**NSP：其实好像没啥用**

SpanBERT 还有一个和原始 BERT 训练很不同的地方，它没用 Next Sentence Prediction (NSP) 任务，而是直接用 Single-Sequence Training，也就是根本不加入 NSP 任务来判断是否两句是上下句，直接用一句来训练。

其实这并不是首次发现 NSP 好像没啥用了，在之前的 XLNet 中就已发现类似结果。关于 XLNet 细节，可参考我们的关于XLNET的教程。

而且在 XLNet 团队最近放出的对比博客里复现的几个 BERT 模型中，没有 NSP 的模型在有些任务反而是最优。

对于为什么 NSP 没有用，这里，SpanBERT 作者们给出了下面两个解释：

1. 相比起两句拼接，一句长句，模型可以获得更长上下文（类似 XLNet 的一部分效果）；
2. 在 NSP 的负例情况下，基于另一个文档的句子来预测词，会给 MLM 任务带来很大噪音。

于是 SpanBERT 就没采用 NSP 任务，直接一句长句，然后 MLM 加上 SBO 任务来进行预训练。

**实验，结论，猜想**

还值得一提的是具体实现部分，看实验表格，会发现 BERT 对比时，除了 Google BERT 还有个神奇的 Our BERT，其实就是对原 BERT 的一些训练过程进行了优化，然后训出的 BERT，也是打破 XLNet 记录的 RoBERTa 的训练方式。实际上 SpanBERT 和 RoBERTa 应该就是一个组做的，关于 RoBERTa 可以参考我们RoBERTa的相关教程。

其中主要训练细节是：

1. 训练时用了 Dynamic Masking 而不是像 BERT 在预处理时做 Mask；
2. 取消 BERT 中随机采样短句的策略；
3. 还有对 Adam 优化器中一些参数改变。

论文中主要对抽取式问答，指代消解，关系抽取，还有大名鼎鼎的 GLUE 做了实验。主要结论如下：

+ SpanBERT 普遍强于 BERT；
+ SpanBERT 尤其在抽取式问答上表现好，这应该与它的预训练目标关系较大；
+ 舍弃掉 NSP 的一段长句训练普遍要比原始 BERT 两段拼接的方式要好。

总体来说，是篇看标题感觉好像不怎么样，但看内容惊喜连连，对 BERT 很多训练细节都进行了分析，让人涨姿势的论文。