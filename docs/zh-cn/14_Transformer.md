## 1.Transformer

<!-- https://zhuanlan.zhihu.com/p/34781297 -->
<!-- https://blog.csdn.net/sxf1061926959/article/details/84979692 -->
<!-- https://blog.csdn.net/qq_28385535/article/details/89081387 -->
<!-- https://blog.csdn.net/qq_28385535/article/details/89081387 -->

<!-- https://blog.csdn.net/u012526436/article/details/86295971 -->
<!-- https://blog.csdn.net/u012526436/article/details/86295971 -->
<!-- https://blog.csdn.net/qq_35169059/article/details/101678207?utm_medium=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase -->

<!-- https://zhuanlan.zhihu.com/p/34781297 -->

<!-- https://blog.csdn.net/longxinchen_ml/article/details/86533005 -->
<!-- https://baijiahao.baidu.com/s?id=1651219987457222196&wfr=spider&for=pc -->
<!-- https://blog.csdn.net/Oscar6280868/article/details/97623488 -->

<!-- B站视频教程 -->
<!-- 李宏毅-Trannsformer:  https://www.bilibili.com/video/BV1J441137V6?from=search&seid=16604948832443915718 -->

<!-- 贪心科技： self-attention & transformer:  https://www.bilibili.com/video/BV1jv41167M2?from=search&seid=16604948832443915718 -->

<!-- 汉语自然语言处理-从零解读碾压循环神经网络的transformer模型(一)-注意力机制-位置编码-attention is all you need:  https://www.bilibili.com/video/BV1P4411F77q?from=search&seid=16604948832443915718-->

<!-- https://www.bilibili.com/video/BV18E411f7yB?from=search&seid=16604948832443915718 -->

<!-- Transformer模型(1) 剥离RNN，保留Attention: https://www.bilibili.com/video/BV1qZ4y1H7Pg?from=search&seid=109824070291948358 -->

<!-- Transformer模型(2) 从Attention层到Transformer网络: https://www.bilibili.com/video/BV1Ap4y1Q7nT?from=search&seid=109824070291948358 -->

<!-- https://github.com/aespresso/a_journey_into_math_of_ml -->

前一段时间谷歌推出的Bert(我们将在其他章节给读者介绍）模型在11项NLP任务中夺得SOTA结果，引爆了整个NLP界。而Bert取得成功的一个关键因素是Transformer的强大作用。谷歌的Transformer模型最早是用于机器翻译任务，当时达到了SOTA效果。Transformer改进了RNN最被人诟病的训练慢的缺点，利用Self-Attention机制实现快速并行。并且Transformer可以增加到非常深的深度，充分发掘DNN模型的特性，提升模型准确率。在本节中，我们将研究Transformer模型，把它掰开揉碎，理解它的工作原理。

Transformer由论文《Attention is All You Need》提出，现在是谷歌云TPU推荐的参考模型。论文相关的Tensorflow的代码可以从GitHub获取，其作为Tensor2Tensor包的一部分。哈佛的NLP团队也实现了一个基于PyTorch的版本，并注释该论文。
在本文中，我们将试图把模型简化一点，并逐一介绍里面的核心概念，希望让普通读者也能轻易理解。

!> Attention is All You Need：https://arxiv.org/abs/1706.03762

### 1.Transformer整体结构

首先将这个模型看成是一个黑箱操作。在机器翻译中，就是输入一种语言，输出另一种语言。

<div align=center>
    <img src="zh-cn/img/transformer/p1.jpg" /> 
</div>

那么拆开这个黑箱，我们可以看到它是由编码组件、解码组件和它们之间的连接组成。

<div align=center>
    <img src="zh-cn/img/transformer/p2.jpg" /> 
</div>

编码组件部分由一堆编码器（encoder）构成（论文中是将6个编码器叠在一起——数字6没有什么神奇之处，你也可以尝试其他数字）。解码组件部分也是由相同数量（与编码器对应）的解码器（decoder）组成的。

<div align=center>
    <img src="zh-cn/img/transformer/p3.jpg" /> 
</div>

<div align=center>
    <img src="zh-cn/img/transformer/p5.jpg" /> 
</div>

<div align=center>
    <img src="zh-cn/img/transformer/p4.png" /> 
</div>

可以看到Transformer整体分为两部分： Decoder和Encoder,两部分均可以重复N次运算，在原论文中`N=6`,上图中绿色的框代表我们下文中详细讲解到的结构，框旁边的数字代表我们将在下文中的第几节(第几部分)讲解该操作;除此之外我们还将详细讲解Transformer训练的损失函数(section 9)和训练的Trick(section 10)。


### 2.Self-Attention

首先，模型需要对输入的数据进行一个Embedding操作，也可以理解为类似word2vec的操作，Enmbedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。

<div align=center>
    <img src="zh-cn/img/transformer/p6.png" /> 
</div>

接下来我们详细看一下self-attention，其思想和attention类似，但是self-attention是Transformer用来将其他相关单词的“理解”转换成我们正在处理的单词的一种思路，我们看个例子：

> The animal didn't cross the street because it was too tired

这里的it到底代表的是animal还是street呢，对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，self-attention就能够让机器把it和animal联系起来，接下来我们看下详细的处理过程。

<div align=center>
    <img src="zh-cn/img/transformer/p17.jpg" /> 
</div>

1.首先，self-attention会计算出三个新的向量，在论文中，向量的维度是`512维`，我们把这三个向量分别称为`Query`、`Key`、`Value`，这三个向量是用embedding向量与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为`（64，512）`注意第二个维度需要和embedding的维度一样，其值在BP的过程中会一直进行更新，得到的这三个向量的维度是`64`低于embedding维度的。

<div align=center>
    <img src="zh-cn/img/transformer/p7.jpg" /> 
</div>

那么`Query`、`Key`、`Value`这三个向量又是什么呢？这三个向量对于attention来说很重要，当你理解了下文后，你将会明白这三个向量扮演者什么的角色。


2.计算self-attention的分数值，该分数值决定了当我们在某个位置encode一个词时，对输入句子的其他部分的关注程度。这个分数值的计算方法是Query与Key做点乘，以下图为例，首先我们需要针对`Thinking`这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即`q1·k1`，然后是针对于第二个词即`q1·k2`

<div align=center>
    <img src="zh-cn/img/transformer/p8.jpg" /> 
</div>

3.接下来，把点乘的结果除以一个常数，这里我们除以8($\sqrt(d))，这个值一般是采用上文提到的矩阵的第一个维度的开方即`64`的开方`8`，当然也可以选择其他的值，然后把得到的结果做一个`softmax`的计算。得到的结果即是每个词对于当前位置的词的相关性大小，当然，当前位置的词相关性肯定会会很大

<div align=center>
    <img src="zh-cn/img/transformer/p9.jpg" /> 
</div>

4.下一步就是把`Value`和`softmax`得到的值进行相乘，并相加，得到的结果即是self-attetion在当前节点的值。

<div align=center>
    <img src="zh-cn/img/transformer/p10.jpg" /> 
</div>

在实际的应用场景，为了提高计算速度，我们采用的是矩阵的方式，直接计算出Query, Key, Value的矩阵，然后把embedding的值与三个矩阵直接相乘，把得到的新矩阵Q与K相乘，乘以一个常数，做softmax操作，最后乘上V矩阵

<div align=center>
    <img src="zh-cn/img/transformer/p11.png" /> 
</div>

`x`矩阵中的每一行对应于输入句子中的一个单词。我们再次看到词嵌入向量 (`512`，或图中的`4`个格子)和`q/k/v`向量(`64`，或图中的`3`个格子)的大小差异。
最后，由于我们处理的是矩阵，我们可以合并为一个公式来计算自注意力层的输出。

<div align=center>
    <img src="zh-cn/img/transformer/p12.png" /> 
</div>

这种通过 `query` 和 `key` 的相似性程度来确定 `value` 的权重分布的方法被称为`scaled dot-product attention`。


### 3.Multi-Head Self-Attention

这篇论文更牛逼的地方是给self-attention加入了另外一个机制，被称为“multi-headed” attention，该机制理解起来很简单，就是说不仅仅只初始化一组`Q、K、V`的矩阵，而是初始化多组，Tranformer是使用了`8组`，所以最后得到的结果是`8个矩阵`。

通过增加一种叫做“多头”注意力（“multi-headed” attention）的机制，论文进一步完善了自注意力层，并在两方面提高了注意力层的性能：

1. 它扩展了模型专注于不同位置的能力。在上面的例子中，虽然每个编码都在`z1`中有或多或少的体现，但是它可能被实际的单词本身所支配。如果我们翻译一个句子，比如`“The animal didn’t cross the street because it was too tired”`，我们会想知道`“it”`指的是哪个词，这时模型的“多头”注意机制会起到作用。

2. 它给出了注意力层的多个“表示子空间”（representation subspaces）。接下来我们将看到，对于“多头”注意机制，我们有多个`查询/键/值`权重矩阵集(Transformer使用`8`个注意力头，因此我们对于每个编码器/解码器有八个矩阵集合)。这些集合中的每一个都是随机初始化的，在训练之后，每个集合都被用来将输入词嵌入(或来自较低编码器/解码器的向量)投影到不同的表示子空间中。

<div align=center>
    <img src="zh-cn/img/transformer/p13.jpg" /> 
</div>

在“多头”注意机制下，我们为每个头保持独立的查询/键/值权重矩阵，从而产生不同的查询/键/值矩阵。和之前一样，我们拿`X`乘以$W^Q/W^K/W^V$矩阵来产生查询/键/值矩阵。

如果我们做与上述相同的自注意力计算，只需8次不同的权重矩阵运算，我们就会得到8个不同的`Z`矩阵。

<div align=center>
    <img src="zh-cn/img/transformer/p14.jpg" /> 
</div>

这给我们留下了一个小的挑战，前馈神经网络没法输入`8个矩阵`呀，这该怎么办呢？所以我们需要一种方式，把`8`个矩阵降为`1`个，首先，我们把`8`个矩阵连在一起，这样会得到一个大的矩阵，再随机初始化一个矩阵和这个组合好的矩阵相乘，最后得到一个最终的矩阵。

<div align=center>
    <img src="zh-cn/img/transformer/p15.jpg" /> 
</div>


这就是multi-headed attention的全部流程了，这里其实已经有很多矩阵了，我们把所有的矩阵放到一张图内看一下总体的流程。

<div align=center>
    <img src="zh-cn/img/transformer/p16.jpg" /> 
</div>

既然我们已经摸到了注意力机制的这么多“头”，那么让我们重温之前的例子，看看我们在例句中编码“it”一词时，不同的注意力“头”集中在哪里：

<div align=center>
    <img src="zh-cn/img/transformer/p18.jpg" /> 
</div>


当我们编码“it”一词时，一个注意力头集中在“animal”上，而另一个则集中在“tired”上，从某种意义上说，模型对“it”一词的表达在某种程度上是“animal”和“tired”的代表。

然而，如果我们把所有的attention都加到图示里，事情就更难解释了：

<div align=center>
    <img src="zh-cn/img/transformer/p19.jpg" /> 
</div>


### 4.Position Encodeing

到目前为止，Transformer模型中还缺少一种解释输入序列中单词顺序的方法。为了处理这个问题，transformer给encoder层和decoder层的输入添加了一个额外的向量Positional Encoding，维度和embedding的维度一样，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离。这个位置向量的具体计算方法有很多种，论文中的计算方法如下

<div align=center>
    <img src="zh-cn/img/transformer/p20.png" /> 
</div>

其中`pos`是指当前词在句子中的位置，`i`是指向量中每个值的`index`，可以看出，在偶数位置，使用正弦编码，在奇数位置，使用余弦编码，这里提供一下代码。

```python
position_encoding = np.array(
    [[pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)] for pos in range(max_seq_len)])

position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

```

最后把这个Positional Encoding与embedding的值相加，作为输入送到下一层。

<div align=center>
    <img src="zh-cn/img/transformer/p21.jpg" /> 
</div>


为了让模型理解单词的顺序，我们添加了位置编码向量，这些向量的值遵循特定的模式。如果我们假设词嵌入的维数为4，则实际的位置编码如下：

<div align=center>
    <img src="zh-cn/img/transformer/p22.jpg" /> 
</div>

尺寸为4的迷你词嵌入位置编码实例,这个模式会是什么样子？在下图中，每一行对应一个词向量的位置编码，所以第一行对应着输入序列的第一个词。每行包含`512`个值，每个值介于`1`和`-1`之间。我们已经对它们进行了颜色编码，所以图案是可见的。

<div align=center>
    <img src="zh-cn/img/transformer/p23.png" /> 
</div>


20字(行)的位置编码实例，词嵌入大小为512(列)。你可以看到它从中间分裂成两半。这是因为左半部分的值由一个函数(使用正弦)生成，而右半部分由另一个函数(使用余弦)生成。然后将它们拼在一起而得到每一个位置编码向量。



### 5.Resdiual,Layer Norm & Feed Forward Layer

**+ Resdiual(残差结构)**

残差结构来源于计算机视觉中的ResNet,如果读者感兴趣可以参考本人在<https://dataxujing.github.io/CNN-paper2/#/zh-cn/chapter1>中的讲解，其结构：

<div align=center>
    <img src="zh-cn/img/transformer/p28.png" /> 
</div>


**+ Layer Norm**

说到normalization，那就肯定得提到 [Batch Normalization](https://terrifyzhao.github.io/2018/02/08/Batch-Normalization%E6%B5%85%E6%9E%90.html)。BN的主要思想就是：在每一层的每一批数据上进行归一化。我们可能会对输入数据进行归一化，但是经过该网络层的作用后，我们的数据已经不再是归一化的了。随着这种情况的发展，数据的偏差越来越大，我的反向传播需要考虑到这些大的偏差，这就迫使我们只能使用较小的学习率来防止梯度消失或者梯度爆炸。

BN的具体做法就是对每一小批数据，在批这个方向上做归一化。如下图所示：


<div align=center>
    <img src="zh-cn/img/transformer/p24.png" /> 
</div>

可以看到，右半边求均值是沿着数据 batch_size的方向进行的，其计算公式如下：

<div align=center>
    <img src="zh-cn/img/transformer/p25.png" /> 
</div>

那么什么是 Layer normalization:<https://arxiv.org/abs/1607.06450> 呢？它也是归一化数据的一种方式，不过 LN 是在每一个样本上计算均值和方差，而不是BN那种在批方向计算均值和方差！

<div align=center>
    <img src="zh-cn/img/transformer/p26.png" /> 
</div>

下面看一下 LN 的公式：

<div align=center>
    <img src="zh-cn/img/transformer/p27.png" /> 
</div>


**+ Feed Forward Layer**

Feed Forward 层比较简单，是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下。

<div align=center>
    <img src="zh-cn/img/transformer/p29.jpg" /> 
</div>

`X`是输入，Feed Forward 最终得到的输出矩阵的维度与 `X` 一致。

如果我们去可视化这些向量以及这个和自注意力相关联的层-归一化操作，那么看起来就像下面这张图描述一样：

<div align=center>
    <img src="zh-cn/img/transformer/p30.jpg" /> 
</div>

### 6.Masked Multi-Head Attention

mask表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。其中，padding mask 在所有的 scaled dot-product attention 里面都需要用到，而 sequence mask 只有在 decoder 的 self-attention 里面用到。

**Padding Mask**

什么是 padding mask 呢？因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。

具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！而我们的 padding mask 实际上是一个张量，每个值都是一个Boolean，值为 False 的地方就是我们要进行处理的地方。

**Sequence Mask**

文章也提到，sequence mask 是为了使得 decoder 不能看见未来的信息。也就是对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。

那么具体怎么做呢？也很简单：产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的。

Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作，因为在翻译的过程中是顺序翻译的，即翻译完第 `i` 个单词，才可以翻译第 `i+1` 个单词。通过 Masked 操作可以防止第 `i` 个单词知道 `i+1` 个单词之后的信息。下面以 `"我有一只猫"` 翻译成 `"I have a cat"` 为例，了解一下 Masked 操作。

下面的描述中使用了类似 Teacher Forcing 的概念，不熟悉 Teacher Forcing 的童鞋可以参考我们在Seq2Seq 模型详解。在 Decoder 的时候，是需要根据之前的翻译，求解当前最有可能的翻译，如下图所示。首先根据输入 "<Begin>" 预测出第一个单词为 "I"，然后根据输入 "<Begin> I" 预测下一个单词 "have"。


<div align=center>
    <img src="zh-cn/img/transformer/p31.jpeg" /> 
</div>

Decoder 可以在训练的过程中使用 Teacher Forcing 并且并行化训练，即将正确的单词序列 `(<Begin> I have a cat)` 和对应输出`(I have a cat <end>)` 传递到 Decoder。那么在预测第 `i` 个输出时，就要将第 `i+1` 之后的单词掩盖住，注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的，下面用 `0 1 2 3 4 5` 分别表示 `"<Begin> I have a cat <end>"`。

第一步：是 Decoder 的输入矩阵和 Mask 矩阵，输入矩阵包含 `"<Begin> I have a cat" (0, 1, 2, 3, 4)` 五个单词的表示向量，Mask 是一个 `5×5`的矩阵。在 Mask 可以发现单词 `0`只能使用单词 `0` 的信息，而单词 `1` 可以使用单词 `0, 1` 的信息，即只能使用之前的信息。

<div align=center>
    <img src="zh-cn/img/transformer/p32.png" /> 
</div>


第二步：接下来的操作和之前的 Self-Attention 一样，通过输入矩阵 `X`计算得到 `Q`, `K`, `V` 矩阵。然后计算 `Q` 和 $K^T$ 的乘积 $QK^T$。

<div align=center>
    <img src="zh-cn/img/transformer/p33.jpeg" /> 
</div>

第三步：在得到$QK^T$ 之后需要进行 Softmax，计算 attention score，我们在 Softmax 之前需要使用 Mask矩阵遮挡住每一个单词之后的信息，遮挡操作如下：

<div align=center>
    <img src="zh-cn/img/transformer/p34.jpeg" /> 
</div>

得到 Mask $QK^T$ 之后在 Mask $QK^T$ 上进行 Softmax，每一行的和都为 1。但是单词 `0` 在单词 `1, 2, 3, 4` 上的 attention score 都为 0。

第四步：使用 Mask $QK^T$ 与矩阵 `V`相乘，得到输出 `Z`，则单词 `1` 的输出向量 `Z1` 是只包含单词 `1` 信息的。

<div align=center>
    <img src="zh-cn/img/transformer/p35.png" /> 
</div>


第五步：通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵 `Zi`，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出 `Zi` 然后计算得到第一个 Multi-Head Attention 的输出 `Z`，`Z`与输入 `X` 维度一样。


### 7.Decoder中的Multi-Head Attention

Decoder block 第二个 Multi-Head Attention 变化不大， 主要的区别在于其中 Self-Attention 的 `K`, `V`矩阵不是使用 上一个 Decoder block 的输出计算的，而是使用 Encoder 的编码信息矩阵 `C` 计算的。

根据 Encoder 的输出 `C`计算得到 `K`, `V`，根据上一个 Decoder block 的输出 `Z` 计算 `Q` (如果是第一个 Decoder block 则使用输入矩阵 `X`进行计算)，后续的计算方法与之前描述的一致。

这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 Mask)。

<div align=center>
    <img src="zh-cn/img/transformer/p36.png" /> 
</div>


### 8.Decoder Output

解码组件最后会输出一个实数向量。我们如何把浮点数变成一个单词？这便是线性变换层要做的工作，它之后就是Softmax层。
线性变换层是一个简单的全连接神经网络，它可以把解码组件产生的向量投射到一个比它大得多的、被称作对数几率（logits）的向量里。

不妨假设我们的模型从训练集中学习一万个不同的英语单词（我们模型的“输出词表”）。因此对数几率向量为一万个单元格长度的向量——每个单元格对应某一个单词的分数。接下来的Softmax 层便会把那些分数变成概率（都为正数、上限1.0）。概率最高的单元格被选中，并且它对应的单词被作为这个时间步的输出。

<div align=center>
    <img src="zh-cn/img/transformer/p37.jpg" /> 
</div>

这张图片从底部以解码器组件产生的输出向量开始。之后它会转化出一个输出单词。

### 9.Loss函数

比如说我们正在训练模型，现在是第一步，一个简单的例子——把`“merci”`翻译为`“thanks”`。这意味着我们想要一个表示单词“thanks”概率分布的输出。但是因为这个模型还没被训练好，所以不太可能现在就出现这个结果。

<div align=center>
    <img src="zh-cn/img/transformer/p38.jpg" /> 
</div>

因为模型的参数（权重）都被随机的生成，（未经训练的）模型产生的概率分布在每个单元格/单词里都赋予了随机的数值。我们可以用真实的输出来比较它，然后用反向传播算法来略微调整所有模型的权重，生成更接近结果的输出。

你会如何比较两个概率分布呢？我们可以简单地用其中一个减去另一个。更多细节请参考交叉熵和KL散度。

+ 交叉熵：https://colah.github.io/posts/2015-09-Visual-Information/
+ KL散度：https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained

但注意到这是一个过于简化的例子。更现实的情况是处理一个句子。例如，输入“je suis étudiant”并期望输出是“i am a student”。那我们就希望我们的模型能够成功地在这些情况下输出概率分布：

每个概率分布被一个以词表大小（我们的例子里是6，但现实情况通常是3000或10000）为宽度的向量所代表。第一个概率分布在与“i”关联的单元格有最高的概率;
第二个概率分布在与“am”关联的单元格有最高的概率; 以此类推，第五个输出的分布表示“”关联的单元格有最高的概率.


<div align=center>
    <img src="zh-cn/img/transformer/p39.jpg" /> 
</div>

依据例子训练模型得到的目标概率分布,在一个足够大的数据集上充分训练后，我们希望模型输出的概率分布看起来像这个样子：

<div align=center>
    <img src="zh-cn/img/transformer/p40.jpg" /> 
</div>


因为这个模型一次只产生一个输出，不妨假设这个模型只选择概率最高的单词，并把剩下的词抛弃。这是其中一种方法（叫贪心解码）。另一个完成这个任务的方法是留住概率最靠高的两个单词（例如I和a），那么在下一步里，跑模型两次：其中一次假设第一个位置输出是单词“I”，而另一次假设第一个位置输出是单词“me”，并且无论哪个版本产生更少的误差，都保留概率最高的两个翻译结果。然后我们为第二和第三个位置重复这一步骤。这个方法被称作集束搜索（beam search）。在我们的例子中，集束宽度是2（因为保留了2个集束的结果，如第一和第二个位置），并且最终也返回两个集束的结果（top_beams也是2）。这些都是可以提前设定的参数。



### 10.训练的Trick

既然我们已经过了一遍完整的transformer的前向传播过程，那我们就可以直观感受一下它的训练过程。在训练过程中，一个未经训练的模型会通过一个完全一样的前向传播。但因为我们用有标记的训练集来训练它，所以我们可以用它的输出去与真实的输出做比较。为了把这个流程可视化，不妨假设我们的输出词汇仅仅包含六个单词：`“a”, “am”, “i”, “thanks”, “student”`以及 `“<EOS>”`（end of sentence的缩写形式）。

<div align=center>
    <img src="zh-cn/img/transformer/p41.png" /> 
</div>

我们模型的输出词表在我们训练之前的预处理流程中就被设定好。

一旦我们定义了我们的输出词表，我们可以使用一个相同宽度的向量来表示我们词汇表中的每一个单词。这也被认为是一个one-hot 编码。所以，我们可以用下面这个向量来表示单词`“am”`：

<div align=center>
    <img src="zh-cn/img/transformer/p42.jpg" /> 
</div>

例子：对我们输出词表的one-hot 编码

训练过程中的一些参数设定：

<div align=center>
    <img src="zh-cn/img/transformer/p43.png" /> 
</div>


### 11.推荐的Code


<div align=center>
    <img src="zh-cn/img/transformer/p44.png" /> 
</div>


<div align=center>
    <img src="zh-cn/img/transformer/p45.png" /> 
</div>

------

## 2.Vanilla Transformer: Character-Level Language Modeling with Deeper Self-Attention

<!-- https://blog.csdn.net/pingpingsunny/article/details/105056297 -->
<!-- https://zhuanlan.zhihu.com/p/84159401 -->
<!-- https://www.cnblogs.com/huangyc/p/11445150.html -->

<!-- Universal Transformer 重新将recurrence引入transformer，并加入自适应的思想，使得transformer图灵完备，并有着更好的泛化性和计算效率 -->

Transformer-XL主要对比的对象即为Vanilla Transformer,这篇Paper的标题为Character-Level Language Modeling with Deeper Self-Attention，按照Transformer-XL对该Paper的称呼，习惯成本节要解读的模型结构为Vanilla Transformer。

语言模型有word-level（词级）和character-level（字符级）等，word-level语言模型通常在词序列的基础上建模，而character-level语言模型通常是在字符序列的基础上建模。Word-level语言模型会遇到OOV（out of vocabulary）问题，即词不在词表中的情况，而character-level语言模型则不会出现此问题

语言模型一般较多使用RNN网络来建模，而character序列比Word序列更长，因此，模型的优化更难。针对此问题，有文献提出将字符序列分割成多段来处理，相邻段之间有信息的前向传递来学习更长期的依赖，但是梯度的反向传播被截断。而Vanilla Transformer也将序列分成多段，不同的是Vanilla Transformer使用Transformer来对字符序列建模，取得了SOTA的成果。

Vanilla Transformer在使用Transformer对字符序列建模，相邻的每段之间没有前向和后向的信息交互，同时增加了辅助损失函数来加速模型的训练。增加的辅助损失函数有3个，一是预测序列中的每个字符，二是在中间层也预测每个字符，三是每次预测多个字符。

先前的语言模型一般是通过一个完整的序列预测最后一个词或字符，而此文则预测每个字符，也可以说是一种seq-to-seq模型，只是没有将整个序列先编码成一个向量再解码成字符，而是将encoder和decoder合二为一，直接预测。


### 1.Vanilla Transformer的结构

首先，作者要解决的问题是字级别的语言模型，相比词级别的语言模型，字符级别语言模型明显需要依赖的距离特别长，比如说一句话某个位置是应该使用she还是he，是依赖于前面的主语情况，这个主语可能距离此单词位置的有十几个单词，每个单词7-8字母长度，那么这就将近100+个字符长度了，作者使用Transformer的结构主要原因是他认为该结构很容易做到在任意距离上的信息传递，而相对RNN（LSTM）这种结构，就需要按照时间一步一步的传递信息，不能做到跨越距离。

这篇文章虽然用到了Transformer结构，但与Attention is all you need(Transformer上一节）是有差异的。原Transformer整体是一个seq2seq结构。而Vanilla Transformer只利用了原Transformer的decode的部分结构，也就是一个带有mask的attention层+一个ff层。

如果将 `"一个带有mask的attention层+一个ff层"` 称为一个layer，那么Vanilla Transformer一共有64个这样的layer，每一个layer有2个head，`model_dim=512`，ff层的`hidden_units=2048`，sequence的长度为`512`。对于训练语言模型来说，这已经是一个很深的网络了，要知道对于大名鼎鼎的BerT网络的层数也就12层（base）和24层（large）了。

另外，之所以使用mask结构是因为语言模型的定义是$p(x_i|x_0,x_1,...,x_{i-1})$，也就是根据前`i`个字符预测第`i+1`个字符，如果你已经提前看到了答案（也就是第`i+1`个字符甚至更后面的字符内容），那就没有预测的意义了，这里加mask与原Transformer的decode部分的带有mask的self-attention道理都是一样的。

Positional Embeddings：RNN结构的网络对于类似于语言模型这种序列性的数据编码带有天然的优势，但缺点就是不能并行，必须要step by step。而attention结构最大的优点就是可以实现并行，但它不能表达序列性，所以为了给网络加入识别序列性就要引入 位置编码 Positional Embeddings。在原Transformer中，位置编码的编码信息是固定的，不需要学习，具体编码方式如下，输出为pos embedding。将`word embedding + pos embedding`整体作为网络的输入，并且仅在第一层加入了位置编码，之后的每层都不会再次加入。而对于Vanilla Transformer，作者认为它的网络深度太深了，如果只在第一层加入pos embedding，那么经过多层传递，这个信息很容易丢失，所以它是每层都会将上一层的输出与pos embedding加在一起作为下一层的输入，而且，pos embedding是可学习的。所以，仅pos embedding模型就要学习 `N\times L\times dim` 个参数，其中`N`是网络的层数（本文`64`层），`L`是上下文的长度(本文`512`)，dim是embedding的维度（`本文=512`）。

```python
# 这个是Transformer的Postion encoding
def positional_encoding(dim, seq_length, dtype=tf.float32):
    """
    :param dim: 编码后的维度
    :param seq_length: 序列的最大长度
    :param dtype:
    :return:
    """
    pos_encode = np.array([pos/np.power(10000, 2*i/dim) for pos in range(seq_length) for i in range(dim)])
    pos_encode[0::2] = np.sin(pos_encode[0::2])
    pos_encode[1::2] = np.cos(pos_encode[1::2])
    return tf.convert_to_tensor(pos_encode.reshape([seq_length, dim]), dtype=dtype, name='positional_encoding')
```

总之，从结构上来说，Vanilla Transformer没有什么太特别的地方，用的组件都是原Transformer这篇论文中用到的，甚至还精简了一些，无非就是Vanilla Transformer的网络深度非常深。这个深度导致在训练的时候很难收敛，个人认为这篇论文中值得学习的就是为了达到收敛目的，作者使用的一些小Trick，这些小Trick对于我们以后解决类似的问题是很有帮助的。


### 2.Vanilla Transformer训练时作者的一些Trick

作者在论文中说当网络的深度超过10的时候，就很难让模型收敛，准确率也很低，所以如果大家训练的网络深度超过10的时候就可以部分借鉴这篇论文中的训练方法：引入辅助的loss。 如下图所示，这个辅助的loss分为3类：Multiple Positions； Intermediate Layer Losses； Multiple Targets

为了方便，我们只以2层来展示，且每一个segment的length=4，原本我们是根据`t0~t3`的输入，在`H`节点这个位置预测`t4`的结果，loss就是`H`节点的输入计算一个交叉熵。现在辅助loss的第一类loss就是：对于最后一层所有的节点都计算下一步应该预测的字符，即在节点`E`处根据输入`t0`，预测输出为`t1`，在节点F处根据输入为`t0`和`t1`，输出是`t2`，以此类推。然后将每一个Positions处的loss加起来。第一类loss贯穿整个train的全部阶段，不发生衰减。

<div align=center>
    <img src="zh-cn/img/transformer/vanilla/p1.png" /> 
</div>

辅助loss的第二类是除了在最后一层计算交叉熵loss之外，在中间层也要计算，即在节点`A`处根据输入`t0`，预测输出为`t1`，以此类推，但中间层的loss并不贯穿整个train始终，而是随着训练进行，逐渐衰减，衰减的方式是，一共有`n`层网络，当训练进行到 `(k/(2*n))`时停止计算第`k`层loss。也就是说当训练进行到一半的时候，所有的 中间层 都不再贡献loss。

<div align=center>
    <img src="zh-cn/img/transformer/vanilla/p2.png" /> 
</div>


辅助loss的第三类是每次预测时所预测几个字符，在本论文中，每次预测下一步和下下步的字符结果，具体的看下面的图即可，非常清楚。但对于下下步的预测结果产生的loss是要发生衰减的，论文中该loss乘以`0.5`后再加入到整体的loss中。

<div align=center>
    <img src="zh-cn/img/transformer/vanilla/p3.png" /> 
</div>


### 3.Vanilla Transformer的相关测试结果


作者使用的数据集有`enwik8`，`lm1b`，`text8`这3个，列举了`64`层的transformer模型与`12`层的transformer模型（目的是比较一下是否深度增加效果更好）还有一些RNN结构的模型进行了比较，实践证明该方法是比较好的，具体数据见论文，此处不列出。

但是作者的消融实验的比较结果我认为是很有意义的，这个对于我们以后设计模型有参考性，就是作者这篇论文里提到了加了3种辅助loss帮助训练，还有就是作者使用了momentum优化器训练，使用的pos embedding也是跟之前不同的。那么这些因素到底有没有用，如果有用，哪个用处大，有多大？针对这个问题作者进行了一个比较，比较的基线是上面讲的`64`层模型。

可以看出，辅助loss中的Multiple Positions和Intermediate Layer Losses效果是最明显的，至于使用了需要学习的pos embedding并没有太大的作用，优化器和Multiple Targets的辅助loss感觉效果都不大。

<div align=center>
    <img src="zh-cn/img/transformer/vanilla/p4.png" /> 
</div>



### 4.其他

Vanilla Transformer是Transformer和Transformer-XL中间过度的一个算法,其结构图如下：

<div align=center>
    <img src="zh-cn/img/transformer/vanilla/p5.png" /> 
</div>

它使用$x_1,x_2,...,x_{n−1}$ 预测字符$x_n$，而$x_n$之后的序列都被mask掉。论文中使用`64`层模型，并仅限于处理 `512`个字符这种相对较短的输入，因此它将输入分成段，并分别从每个段（segment）中进行学习，如上图所示。 在测试阶段如需处理较长的输入，该模型会在每一步中将输入向右移动一个字符，以此实现对单个字符的预测。

简单说：
+ 训练阶段，预测下一个字符的时候，并不是用所有上下文，只是用一个段（segment）的上下文预测。以此训练整个网络。
+ 测试阶段，也是用一个段（segment）大小的context预测下一个字符。在预测的时候，会对固定长度的segment做计算，一般取最后一个位置的隐向量作为输出。为了充分利用上下文关系，在每做完一次预测之后，就对整个序列向右移动一个位置，再做一次计算，如上图所示，这导致计算效率非常低。

Vanilla Transformer的三个缺点：
+ 上下文长度受限：字符之间的最大依赖距离受输入长度的限制，模型看不到出现在几个句子之前的单词。
+ 上下文碎片：对于长度超过512个字符的文本，都是从头开始单独训练的。段与段之间没有上下文依赖性，会让训练效率低下，也会影响模型的性能。
+ 推理速度慢：在测试阶段，每次预测下一个单词，都需要重新构建一遍上下文，并从头开始计算，这样的计算速度非常慢。

下一节，我们将看到Transformer-XL是如何解决这些问题的！

------

## 3.Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

<!-- https://www.cnblogs.com/huangyc/p/11445150.html -->
<!-- https://zhuanlan.zhihu.com/p/84159401 -->

Transformer-XL架构在vanilla Transformer的基础上引入了两点创新：

+ 循环机制（Recurrence Mechanism）
+ 相对位置编码（Relative Positional Encoding）。

以克服Vanilla Transformer的缺点。与Vanilla Transformer相比，Transformer-XL的另一个优势是它可以被用于单词级和字符级的语言建模。

### 1.Segment-Level Recurrence

<div align=center>
    <img src="zh-cn/img/transformer/xl/p1.png" /> 
</div>

为了解决Vanilla Transformer提到的问题，在Transformer的基础上，Transformer-XL提出了一个改进，在对当前Segment进行处理的时候，缓存并利用上一个Segment中所有layer的隐向量序列，而且上一个Segment的所有隐向量序列只参与前向计算，不再进行反向传播，这就是所谓的segment-level Recurrence。

我们详细看一下如何操作。Transformer本身是可以设置multi-heads，但是在后文中为了简化描述采用单个head。将两个连续的segments表示为
$$s_\tau=[x_{\tau,1},x_{\tau,2},...,x_{\tau,L}]$$
$$s_{\tau+1}=[x_{\tau+1,1},x_{\tau+1,2},...,x_{\tau+1,L}]$$

`L`是序列长度。假设整个模型中，包含`N`层Transformer，那么每个Segment中就有`N`组长度为`L`的隐向量序列, 将第$\tau$个Segment的第`n`层隐向量序列表示为$h^{n}_ {\tau}\in R^{L\times d}$，`d`是隐向量的维度。那么第$\tau+1$个segment的第`n`层隐向量序列，可以由下面的一组公式计算得出。

<div align=center>
    <img src="zh-cn/img/transformer/xl/p2.png" /> 
</div>

这里$SG(.)$是stop-gradinet,不再对$s_\tau$的隐含向量做反向传播。$\tilde{h}^{n-1}_ {\tau+1}$是对两个银行向量序列眼长度方向的拼接，$[ ]$内两个隐含向量的维度都是$L \times d$，拼接之后的向量维度是$2L\times d$。3个$W$分别对应`query`，`key`和`value`的转化矩阵。注意`q`的计算方式不变，只使用当前segment中的隐向量，计算得到的`q`序列长度仍然是`L`。 `k`和`v`采用拼接之后的$\tilde{h}$来计算，计算出来的序列长度是`2L`。 之后的计算就是标准的Transformer计算。计算出来的第`n`层隐向量序列长度仍然是`L`，而不是`2L`。Transformer的输出隐向量序列长度取决于`query`的序列长度，而不是`key`和`value`。

训练和预测过程如上图所示。这张图上有一个点需要注意，在当前segment中，第`n`层的每个隐向量的计算，都是利用下一层中包括当前位置在内的，连续前`L`个长度的隐向量，这是在上面的公式组中没有体现出来的，也是文中没有明说的。 每一个位置的隐向量，除了自己的位置，都跟下一层中前`(L-1)`个位置的`token`存在依赖关系，而且每往下走一层，依赖关系长度会增加`(L-1)`，如上图中Evaluation phase所示，所以最长的依赖关系长度是`N(L-1)`，`N`是模型中layer的数量。 `N`通常要比`L`小很多，比如在BERT中，`N=12`或者`24`，`L=512`，依赖关系长度可以近似为$O(N\times L)$。
在对长文本进行计算的时候，可以缓存上一个segment的隐向量的结果，不必重复计算，大幅提高计算效率。

上文中，我们只保存了上一个segment，实际操作的时候，可以保存尽可能多的segments，只要内存或者显存放得下。论文中的试验在训练的时候，只缓存一个segment，在预测的时候，会缓存多个segments。

### 2.Relative Position Encodings

在Vanilla Transformer中，为了表示序列中`token`的顺序关系，在模型的输入端，对每个`token`的输入embedding，加一个位置embedding。位置编码embedding或者采用正弦\余弦函数来生成，或者通过学习得到。在Transformer-XL中，这种方法行不通，每个segment都添加相同的位置编码，多个segments之间无法区分位置关系。Transformer-XL放弃使用绝对位置编码，而是采用相对位置编码，在计算当前位置隐向量的时候，考虑与之依赖`token`的相对位置关系。具体操作是，在算attention score的时候，只考虑`query`向量与`key`向量的相对位置关系，并且将这种相对位置关系，加入到每一层Transformer的attention的计算中。

我们对两种方法做个对比。下面一组公式是Vanilla Transformer计算attention的方式，$E_x$表示`token`的输入embedding，$U$是绝对位置编码embedding，两个$W$分别是`query`矩阵和`key`矩阵。下面的公式是对$(E_{x_i}+U_i)W_qW_k(E_{x_j}+U_j)$作了分解

<div align=center>
    <img src="zh-cn/img/transformer/xl/p3.png" /> 
</div>


下面一组公式，是Transformer-XL计算attention的方式。首先，将绝对位置编码$U$，替换成了相对位置编$R_{i-j}$。 插一句，因为$i$只利用之前的序列，所以$i-j>=0$。其次,对于所依赖的`key`向量序列，query向量$U_iW_q$都是固定的,因此将上面公式组(c)中的$U_iW_q$替换为$u\in R^d$,将上面中的(d)中的$U_iW_q$替换为$v\in R^d$, $u$和$v$都通过学习得到。最后我们再将$W_k$矩阵再细分成两组矩阵$W_{k,E}$和$W_{k,R}$，分别生成基于内容的`key`向量和基于位置的`key`向量。可以仔细思考一下每一项中的依赖关系。

<div align=center>
    <img src="zh-cn/img/transformer/xl/p4.png" /> 
</div>


相对位置关系用一个位置编码矩阵$R\in R^{L_{max} \times d}$来表示，第$i$行表示相对位置间隔为$i$的位置向量。论文中强调$R$采用正弦函数生成，而不是通过学习得到的，好处是预测时，可以使用比训练距离更长的位置向量。

最后来看一下Transformer-XL的完整计算公式，如下所示，只有前3行与Vanilla Transformer不同，后3行是一样的。第3行公式中，计算`A`的时候直接采用`query`向量，而不再使用$E_xW_q$表示。 最后需要注意的是，每一层在计算attention的时候，都要包含相对位置编码。

<div align=center>
    <img src="zh-cn/img/transformer/xl/p5.png" /> 
</div>


### 3.其他

**优点** 

1. 在几种不同的数据集（大/小，字符级别/单词级别等）均实现了SOTA的语言建模结果。
2. 结合了深度学习的两个重要概念——循环机制和注意力机制，允许模型学习长期依赖性，且可能可以扩展到需要该能力的其他深度学习领域，例如音频分析（如每秒16k样本的语音数据）等。
3. 在inference阶段非常快，比之前最先进的利用Transformer模型进行语言建模的方法快300～1800倍。

**不足**

1. 尚未在具体的NLP任务如情感分析、QA等上应用。
没有给出与其他的基于Transformer的模型，如BERT等，对比有何优势。
2. 训练模型需要用到大量的TPU资源。


------

最后我们详细介绍Transformer在目标检测(计算机视觉领域)的应用。

## 4.Detr: End-to-End Object Detection with Transformers