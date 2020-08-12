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

---

## 2.Transformer-XL