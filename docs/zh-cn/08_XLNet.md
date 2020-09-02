## XLNet: Generalized Autoregressive Pretraining for Language Understanding

<!-- https://blog.csdn.net/weixin_37947156/article/details/93035607 -->
<!-- https://wmathor.com/index.php/archives/1475/ -->

<!-- https://zhuanlan.zhihu.com/p/70257427 -->
<!-- http://fancyerii.github.io/2019/06/30/xlnet-theory/ -->
<!-- http://fancyerii.github.io/2019/07/20/xlnet-codes3/#positional_embedding -->
<!-- https://blog.csdn.net/Magical_Bubble/article/details/89060213 -->
<!-- https://zhuanlan.zhihu.com/p/71916499 -->
<!-- 张俊林老师--讲的比较透彻） XLNet:运行机制及和Bert的异同比较： https://zhuanlan.zhihu.com/p/70257427 -->

<!-- 官方XLNet GitHub地址： https://github.com/zihangdai/xlnet -->

<!-- https://www.bilibili.com/video/BV1zJ411P7X6?from=search&seid=4866638900620705938 -->
<!-- https://www.bilibili.com/video/BV1df4y1y7Mp?from=search&seid=4866638900620705938 -->
<!-- https://www.bilibili.com/video/BV1y64y1c78H?from=search&seid=4866638900620705938 -->


### 1.摘要

2018年，谷歌发布了基于双向Transformer的大规模预训练语言模型BERT，刷新了11项NLP任务的最优性能记录，为NLP 领域带来了极大的惊喜。很快，BERT就在圈内普及开来，也陆续出现了很多与它相关的新工作。

BERT带来的震撼还未平息，来自卡耐基梅隆大学与谷歌大脑的研究者又提出新型预训练语言模型XLNet，在 SQuAD、GLUE、RACE等20个任务上全面超越 BERT。

作者表示，BERT 这样基于去噪自编码器的预训练模型可以很好地建模双向语境信息，性能优于基于自回归语言模型的预训练方法。然而，由于需要mask一部分输入，BERT忽略了被mask 位置之间的依赖关系，因此出现预训练和微调效果的差异（pretrain-finetune discrepancy）。

基于这些优缺点，该研究提出了一种泛化的**自回归预训练模型XLNet**。 XLNet 可以：

1）通过最大化所有可能的因式分解顺序的对数似然，学习双向语境信息；
2）用自回归本身的特点克服BERT的缺点。此外，XLNet还融合了当前最优自回归模型Transformer-XL的思路

最终，XLNet 在20个任务上超过了BERT的表现，并在18 个任务上取得了当前最佳效果（state-of-the-art），包括机器问答、自然语言推断、情感分析和文档排序。

以前超越BERT的模型很多都在它的基础上做一些修改，本质上模型架构和任务都没有太大变化。但是在这篇新论文中，作者从自回归（autoregressive）和自编码（autoencoding）两大范式分析了当前的预训练语言模型，并发现它们虽然各自都有优势，但也都有难以解决的困难。为此，研究者提出XLNet，并希望结合大阵营的优秀属性。

### 2.AR(AutoRegressive)与AE(AutoEncoder LM)两大阵营

<div align=center>
    <img src="zh-cn/img/xlnet/p1.png" /> 
</div>


#### 1.自回归语言模型（AutoRegressive LM）

在 ELMO／BERT 出来之前，大家通常讲的语言模型其实是根据上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行（就是根据下文预测前面的单词）。这种类型的LM 被称为自回归语言模型。[GPT-1](https://github.com/huggingface/pytorch-openai-transformer-lm#pytorch-implementation-of-openais-finetuned-transformer-language-model),[GPT-2](https://github.com/graykode/gpt-2-Pytorch#gpt2-pytorch-with-text-generator)就是典型的自回归语言模型。ELMO 尽管看上去利用了上文，也利用了下文，但是本质上仍然是自回归 LM，这个跟模型具体怎么实现有关系。ELMO 是分别做了两个方向的自回归 LM（从左到右以及从右到左两个方向的语言模型），然后把 LSTM 的两个方向的隐状态拼接到一起，来体现双向语言模型这个事情的。所以其本质上仍然是自回归语言模型。

AR语言模型是一种使用上下文词来预测下一个词的模型。但是在这里，上下文单词被限制在两个方向，前向或后向。

<div align=center>
    <img src="zh-cn/img/xlnet/p2.png" /> 
</div>

+ AR语言模型的优势是擅长生成式自然语言处理任务。 因为在生成上下文时，通常是前向的。AR 语言模型很自然地适用于此类NLP任务。它的优点跟下游 NLP 任务有关，比如生成类 NLP 任务，比如文本摘要，机器翻译等，在实际生成内容的时候，就是从左向右的，自回归语言模型天然匹配这个过程
+ 但AR语言模型有一些缺点，它只能使用前向上下文或后向上下文，这意味着它不能同时使用前向和后向上下文。

给定文本序列$x=[x_1,…,x_T]$，语言模型的目标是调整参数使得训练数据上的似然函数最大：
$$\max_\theta\log p_{\theta}(x)=\sum_{t=1}^{T}\log p_{\theta}(x_t|x_{< t})=\sum_{t=1}^{T}\log \frac{\exp(h_{\theta}(x_{1:t-1})^Te(x_t))}{\sum_x^{'}\exp(h_{\theta}(x_{1:t-1})^Te(x^{'}))}$$

记号$x< t$表示$t$时刻之前的所有$x$，也就是$x_{1:t−1}$。$h_{\theta}(x_{1:t−1})$是RNN或者 Transformer（注：Transformer 也可以用于语言模型，比如在 OpenAI GPT）编码的$t$时刻之前的隐状态。$e(x)$是词 $x$的embedding.


#### 2.自编码语言模型（AutoEncoder LM）

<div align=center>
    <img src="zh-cn/img/xlnet/p3.png" /> 
</div>

BERT 通过将序列$x$中随机挑选`15%`的Token变成 `[MASK]`得到带噪声版本的  
$\hat{x}$ 。假设被Mask的原始值为$\bar{x}$，那么BERT希望尽量根据上下文恢复（猜测）出原始值，也就是：

$$\max_{\theta}\log p_{\theta}(\bar{x}|\hat{x})\approx \sum_{t=1}^{T}m_t\log p_{\theta}(x_t|\hat{x})=\sum_{t=1}^Tm_t\log \frac{\exp(H_{\theta}(x)_ t^Te(x_t))}{\sum_{x^{'}}\exp(H_{\theta}(x)_ t^Te(x^{'}))}$$

上式中，若$m_t=1$表示$t$时刻是一个`Mask`，需要恢复。$H_{\theta}$是一个Transformer，它把长度为`T`的序列$x$ 映射为隐状态的序列$H_{\theta}(x)=[H_{\theta}(x)_ 1,H_{\theta}(x)_ 2,...,H_{\theta}(x)_ T]$。 注意：前面的语言模型的RNN在$t$ 时刻只能看到之前的时刻，因此记号是$h_{\theta}(x_{1:t−1})$；而BERT的Transformer（不同与用于语言模型的 Transformer）可以同时看到整个句子的所有 Token，因此记号是 $H_{\theta}(x)$.

这种AE LM的优缺点正好和AR LM反过来，它能比较自然地融入双向语言模型，同时看到被预测单词的上文和下文，这是好处。缺点是啥呢？主要在输入侧引入`[Mask]`标记，导致预训练阶段和 Fine-tuning 阶段不一致的问题，因为 Fine-tuning 阶段是看不到 `[Mask]`标记的

XLNet的出发点就是：能否融合自回归LM和DAE LM 两者的优点。具体来说就是，站在AR 的角度，如何引入和双向语言模型等价的效果。


### 3.排列语言建模（Permutation Language Modeling）

作者们发现，只要在 AR 以及 AE 方式中再加入一个步骤，就能够完美地将两者统一起来，那就是`Permutation`。从上面的比较可以得出，AR 语言建模和 BERT 拥有其自身独特的优势。我们自然要问，是否存在一种预训练目标函数可以取二者之长，同时又克服二者的缺点呢？

作者提出了一种序列语言建模目标，它不仅可以保留 AR 模型的优点，同时也允许模型捕获双向语境。具体来说，一个长度为$T$的序列$x$拥有$T!$种不同的排序方式，可以执行有效的自回归因式分解。从直觉上来看，如果模型参数在所有因式分解顺序中共享，那么预计模型将学习从两边的所有位置上收集信息。

为了提供一个完整的概览图，研究者展示了一个在给定相同输入序列$x$（但因式分解顺序不同）时预测`token` $x_3$ 的示例，如下图所示：比如图的左上，对应的分解方式是3→2→4→1，因此预测$x_3$是不能attend to任何其它词，只能根据之前的隐状态mem来预测。而对于左下，$x_3$可以attend to其它3个词。

<div align=center>
    <img src="zh-cn/img/xlnet/p4.png" /> 
</div>

给定长度为$T$的序列$x$，总共有$T!$种排列方法，也就对应$T!$种链式分解方法。比如假设$x=x_1x_2x_3$，那么总共用$3!=6$种分解方法：

<div align=center>
    <img src="zh-cn/img/xlnet/p5.png" /> 
</div>

注意$p(x_2|x_1x_3)$指的是第一个词是$x_1$并且第三个词是$x_3$的条件下第二个词是$x_2$的概率，也就是说原来词的顺序是保持的。如果理解为第一个词是$x_1$并且第二个词是$x_3$的条件下第三个词是$x_2$，那么就不对了。

如果我们的语言模型遍历$T!$种分解方法，并且这个模型的参数是共享的，那么这个模型应该就能(必须)学习到各种上下文。普通的从左到右或者从右往左的语言模型只能学习一种方向的依赖关系，比如先”猜”一个词，然后根据第一个词”猜”第二个词，根据前两个词”猜”第三个词，...。而排列语言模型会学习各种顺序的猜测方法，比如上面的最后一个式子对应的顺序3→1→2，它是先”猜”第三个词，然后根据第三个词猜测第一个词，最后根据第一个和第三个词猜测第二个词。

因此我们可以遍历$T!$种路径，然后学习语言模型的参数，但是这个计算量非常大($10!=3628800$,10个词的句子就有这么多种组合)。因此实际我们只能随机的采样$T!$里的部分排列，为了用数学语言描述，我们引入几个记号。$Z^T$表示长度为$T$的序列的所有排列组成的集合，则$z\in Z^T$是一种排列方法。我们用$z_t$表示排列的第$t$个元素，而$z< t$表示$z$的第`1`到第`t-1`个元素。

举个例子，假设$T=3$，那么$Z^T$共有6个元素，我们假设其中之一$z=[1,3,2]$，则$z_3=2$，而$z<3=[1,3]$。

有了上面的记号，则排列语言模型的目标是调整模型参数使得下面的似然概率最大：
$$\max_{\theta}E_{z\sim Z_T}[\sum_{t=1}^{T}\log p_{\theta}(x_{z_t}|x_{z< t})]$$

上面的公式看起来有点复杂，细读起来其实很简单：从所有的排列中采样一种，然后根据这个排列来分解联合概率成条件概率的乘积，然后加起来。

注意：上面的模型只会遍历概率的分解顺序，并不会改变原始词的顺序。实现是通过Attention的Mask来对应不同的分解方法。比如$p(x_1|x_3)p(x_2|x_1x_3)p(x_3)$，我们可以在用Transformer编码$x_1$时候让它可以Attend to $x_3$，而把$x_2$`Mask`掉；编码$x_3$的时候把$x_1,x_2$都`Mask`掉。

<div align=center>
    <img src="zh-cn/img/xlnet/p6.png" /> 
</div>

**Permutation Language Modeling具体的实现方式：**

通过随机取一句话排列的一种，然后将末尾一定量的词给 “Mask”（和 BERT里的直接替换"[MASK]" 有些不同）掉，最后用 AR的方式来按照这种排列方式依此预测被“Mask” 掉的词

<div align=center>
    <img src="zh-cn/img/xlnet/p7.png" /> 
</div>


这里我稍微解释下，为什么是 "Mask"掉末尾的一些词，以及随机打乱句子的顺序有什么用？输入句子正常的顺序是 "1→2→3→4→5→6→7"，常规的自回归LM无法同时考虑上下文信息。如果能够同时考虑上下文信息，那 "3" 这个词，需要有 "1→2→4→5→6→7" 这些信息，换句话说，在预测 "3" 之前，我们需要保证模型已经看过 "1→2→4→5→6→7"（无所谓顺序）。而打乱句子的顺序之后（比方说上图的例子），"3"这个词就来到了句子的末尾，此时按照自回归 LM 预测 "3" 的时候，模型已经看过了 "1→2→4→5→6→7"，由此便考虑到了 "3" 的上下文信息。当然，句子到底怎么打乱是无所谓的，因为我们的目标不是具体要预测哪个词，而是谁在最后，就预测谁。

这里再谈一个有意思的点，到底该挑选最后几个做遮掩呢？作者这里设了一个超参数$K$，$K$ 等于总长度除以需要预测的个数。拿上面的例子，总长为$7$而需要预测为$2$，于是$K=7/2$。而论文中实验得出的最佳$K$ 值介于$6$和$7$（更好）之间，其实如果我们取$K$的倒数（即$\frac{1}{6},\frac{1}{7}$），然后转为百分比，就会发现最佳的比值介于$14.3\%$到$16.7\%$之间，还记得BERT论文的同学肯定就会开始觉得眼熟了。因为 BERT 里将 Token 遮掩成 “[MASK]” 的百分比就是$15\%$，正好介于它们之间，我想这并不只是偶然，肯定有更深层的联系。


### 4.没有目标 (target) 位置信息的问题

上面的思想很简单，但是如果我们使用标准的 Transformer 实现时会有问题。下面举个例子

假设输入的句子是”I like New York”，并且一种排列为$z=[1, 3, 4, 2]$，假设我们需要预测的是$z_3=4$，那么根据 Simple LM 的公式：

$$p_{\theta}(X_{z_3}=x|x_{z_1z_2})=p_{\theta}(X_4=x|x_1x_3)=\frac{\exp(e(x)^Th_{\theta}(x_1x_3))}{\sum_{x^{'}}\exp(e(x^{'})^Th_{\theta}(x_1x_3))}$$

我们通常用大写的$X$表示随机变量，比如$X_4$，而小写的$x$表示某一个具体取值，比如假设$x$是 "York"，则$p_{\theta}(X_4=x)$表示第 4个词是"York"的概率。用自然语言描述：$p_{\theta}(X_4=x|x_1x_3)$表示的是第一个词是"I"，第3个词是"New"的条件下第4个词是"York"的概率。

另外我们再假设一种排列为 $z^{’}=[1,3,2,4]$，我们需要预测$z_3=2$，那么：
$$p_{\theta}(X_{z_3}=x|x_{z_1z_2})=p_{\theta}(X_2=x|x_1x_3)=\frac{\exp(e(x)^Th_{\theta}(x_1x_3))}{\sum_{x^{'}}\exp(e(x^{'})^Th_{\theta}(x_1x_3))}$$

我们先不管预测的真实值是什么，先假设$x$是 "York" 时的概率，则$p_{\theta}(X_2=x|x_1x_3)$表示的是第一个词是"I"，第 3 个词是 "New" 的条件下第 2 个词是 "York" 的概率。

我们仔细对比一下上面两个公式会发现它们是相等的。但是根据经验，显然这两个概率是不同的，而且上面的那个概率大一些，因为 York 跟在 New 之后是一个城市，而”York New” 是什么呢？

上面问题的关键是模型并不知道要预测的那个词在原始序列中的位置。了解 Transformer 的读者可能会问：不是输入了位置编码吗？位置编码的信息不能起作用吗？注意：位置编码是和输入的 Embedding 加到一起作为输入的，因此 $p_{\theta}(X_4=x|x_1x_3)$里的 $x_1,x_3$是带了位置信息的，模型（可能）知道（根据输入的向量猜测）"I" 是第一个词，而 New 是第三个词，但是第四个词的向量显然还不知道（知道了就不用预测了），因此就不可能知道它要预测的词到底是哪个位置的词，所以我们必须 "显式" 的告诉模型我要预测哪个位置的词。

为了后面的描述，我们再把上面的两个公式写出更加一般的形式。给定排列$z$，我们需要计算$p_{\theta}(X_{z_t}|x_{z< t}=x)$，如果我们使用普通的 Transformer，那么计算公式为：
$$p_{\theta}(X_{z_t}=x|x_{z< t})=\frac{\exp(e(x)^Th_{\theta}(x_{z< t}))}{\sum_{x^{'}}\exp(e(x^{'})^Th_{\theta}(x_{z< t}))}$$

根据前面的讨论，我们知道问题的关键是模型并不知道要预测的到底是哪个位置的词，为了解决这个问题，我们把预测的位置$z_t$放到模型里：
$$p_{\theta}(X_{z_t}=x|x_{z< t})=\frac{\exp(e(x)^T g_{\theta}(x_{z< t},z_t))}{\sum_{x^{'}}\exp(e(x^{'})^T g_{\theta}(x_{z< t},z_t))}$$

上式中$g_{\theta}(x_{z< t},z_t)$表示这是一个新的模型$g$，并且它的参数除了之前的词$x_{z< t}$，还有要预测的词的位置$z_t$。

### 5.Two-Stream Self-Attention

接下来的问题是用什么模型来表示$g_{\theta}(x_{z< t},z_t)$。当然有很多种可选的函数（模型），我们需要利用 $x_{z< t}$，通过 Attention 机制提取需要的信息，然后预测 $z_t$ 位置的词。那么它需要满足如下两点要求：

+ 为了预测 $x_{z_t}$，$g_{\theta}(x_{z< t},z_t)$ 只能使用位置信息 $z_t$ 而不能使用 $x_{z_t}$。这是显然的：你预测一个词当然不能知道要预测的是什么词
+ 为了预测 $z_t$ 之后的词，$g_{\theta}(x_{z< t},z_t)$ 必须编码了 $x_{z_t}$ 的信息（语义）

但是上面两点要求对于普通的 Transformer 来说是矛盾的无法满足的。这里非常重要，所以我这里再啰嗦一点举一个例子.

假设输入的句子还是”I like New York”，并且一种排列为 $z=[1,3,4,2]$，假设 $t=2$（即 $z_t=z_2=3$），我们现在要计算 $g_{\theta}(x_{z< t},z_t)$，也就是给定第一个位置的词为 "I"，预测第三个位置为 "New" 的概率。显然我们不能使用 "New" 本身的信息，而只能根据第一个位置的 "I" 来预测。假设我们非常幸运的找到了一很好的函数 $g$，它可以能够比较好的预测这个概率 $g_{\theta}(x_1,z_2)$。现在我们轮到计算 $t=3$（即$ z_3=4$），也就是根据 $g_{\theta}(x_1,z_2)$ 和 $z_t$ 来预测 "York"。显然，知道第三个位置是 "New" 对于预测第四个位置是 "York" 会非常有帮助，但是 $g_{\theta}(x_1,z_2)$ 并没有 New 这个词的信息。读者可能会问：你不是说 $g$ 可以比较好的根据第一个词 "I" 预测第三个词 "New" 的概率吗？这里有两点："I" 后面出现 "New" 的概率并不高；在预测 "York" 时我们是知道第三个位置是 New 的，只不过由于模型的限制，我们无法重复利用这个信息。

为了解决这个问题，论文引入了two Stream，也就是两个隐状态：

+ 内容隐状态 $h_{\theta}(x_{z< t})$，简写为 $h_{z_t}$，它就和标准的 Transformer 一样，既编码上下文（context）也编码 $x_{z_t}$ 的内容
+ 查询隐状态 $g_{\theta}(x_{z< t},z_t)$，简写为 $g_{z_t}$，它只编码上下文和要预测的位置 $z_t$，但是不包含 $x_{z_t}$

下面我们介绍一下计算过程。我们首先把查询隐状态 $g^{(0)}_ i$初始化为一个变量$w$，把内容隐状态 $h^{(0)}_ i$初始化为词的 Embedding $e(x_i)$。这里的上标 0 表示第 0 层（不存在的层，用于计算第一层）。因为内容隐状态可以编码当前词，因此初始化为词的 Embedding 是比较合适的.

接着从 $m=1$ 一直到第 $M$ 层，逐层计算：
$$g^{(m)}_ {z_t}\leftarrow Attention(Q=g^{(m-1)}_ {z_t},KV=h^{(m-1)}_ {\color{red}{z< t}};\theta)$$
$$h^{(m)}_ {z_t}\leftarrow Attention(Q=h^{(m-1)}_ {z_t},KV=h^{(m-1)}_ {\color{red}{z<= t}};\theta)$$

+ Query Stream: use $z_t$ but cannot see $x_{z_t}$
+ Content Stream: use both $z_t$ and $x_{z_t}$

上面两个流分别使用自己的 Query 向量 $g_{z_t}$ 和 Content 向量 $h_{z_t}$；但是 Key 和 Value 向量都是用的 $h$。但是注意 Query 流不能访问 $z_t$ 的内容，因此 K 和 V 是 $h^{(m-1)}_ {z< t}$。而 Content 流的 KV 是 $h^{(m-1)}_ {z<= t}$，它包含 $x_{z_t}$

上面的梯度更新和标准的 Self Attention 是一样的。在 Fine-tuning 的时候，我们可以丢弃掉 Query 流而只用 Content 流。最后在计算公式的时候我们可以用最上面一层的 Query 向量 $g^{(M)}_ {z_t}$. 我们可以通过下图来直观的了解计算过程:

<div align=center>
    <img src="zh-cn/img/xlnet/p8.png" /> 
</div>

左上图是 Content 流的计算，假设排列为 3→2→4→1，并且我们现在预测第 1 个位置的词的概率。根据排列，我们可以参考所有 4 个词的 Content，因此 $K\&V=[h^{(0)}_ 1,h^{(0)}_ 2,h^{(0)}_ 3,h^{(0)}_ 4]$,而$Q=h^{(0)}_ 1$; 左下图是 Query 流的计算，因为不能参考自己的内容，因此 $K\&V=[h^{(0)}_ 2,h^{(0)}_ 3,h^{(0)}_ 4]$,而$Q=g^{(0)}_ 1$.

图的右边是完整的计算过程，我们从下往上看。首先 $h$ 和 $g$ 分别被初始化为 $e(x_i)$ 和 $W$，然后 Content Mask 和 Query Mask 计算第一层的输出 $h^{(1)}$ 和 $g^{(1)}$，然后计算第二层……。注意最右边的两个 Mask，我们先看 Content Mask。它的第一行全是红点，表示第一个词可以 attend to 所有的词（根据 3→2→4→1），第二个词可以 attend to 它自己和第三个词……。而 Query Mask 和 Content Mask 的区别就是不能 attend to 自己，因此对角线都是白点。

到此为止，XLNet 的核心思想已经比较清楚了。主要使用 LM，但是为了解决上下文的问题，引入了 Permutation LM。Permutation LM 在预测时需要 target 的位置信息，因此通过引入 Two-Stream，Content 流编码到当前时刻的内容，而 Query 流只参考之前的历史以及当前要预测位置。最后为了解决计算量过大的问题，对于一个句子，我们只预测后$\frac{1}{K}$个词。

接下来 XLNet 借鉴了 Transformer-XL 的优点，它对于很长的上下文的处理是要优于传统的 Transformer 的。我这里只是简单的介绍 Transformer-XL，有兴趣的读者可以参考 [Transformer-XL](https://arxiv.org/abs/1901.02860) 论文或我们关于在Transformer中对Transformer-XL的介绍。


### 6.融入Transformer-XL的理念

**1.Segment Recurrence Mechanism**

尽管 Transformer 最初是为翻译任务而构建的，但最近的趋势表明，它在语言建模上的应用也可以带来显著的效果。但是，为了获得最佳应用，需要对其架构进行一些修改。

为什么？Transformer 有什么问题？与 RNN 相比，Transformer 的一项重大改进是其捕获长期依赖关系的能力。但是，Transformer 需要存储的中间步骤（梯度）信息比 RNN 要多的多，并且随着序列长度的增加而增加。换句话说，如果你试图一次输入整个文档，内存可能会爆炸（BOOOOM！）。

为了防止出现此问题，早期有些做法是将文档分成固定大小的文本段（Segment），一次训练一段。这虽然解决了内存问题，但是破坏了模型捕获长期依赖关系的能力。例如句子 "The daughter had a nice umbrella | that her mother gave her"，如果 "daughter" 和 "her" 属于不同段。那么在编码 "her 时将无法知晓"daughter" 的信息

如何解决这个问题呢？下面就轮到 Transformer-XL 出场了！

Transformer-XL 的重要组件之一，Segment Recurrence Mechanism（段循环机制）想做的就是，能不能在前一段计算完后，将它计算出的隐状态都保存下来，存到一个 Memeory 中，之后在计算当前段的时候，将之前存下来的隐状态和当前段的隐状态拼起来，作为 Attention 机制的 K 和 V，从而获得更长的上下文信息

<div align=center>
    <img src="zh-cn/img/xlnet/p9.png" /> 
</div>

根据之前的思路，我们用 cache 缓存部分历史的状态。计算梯度的时候只使用本 segment 的信息，但是在 forward 的时候其实用到了之前的 segment（甚至很久以前的 segment）的信息，因此它又有点类似于 RNN。下面我们用数学语言来描述状态重用的过程。假设两个相邻的 segment 为$s_{\tau}=[x_{\tau,1},x_{\tau,2},...,x_{\tau,L}]$和$s_{\tau+1}=[x_{\tau+1,1},x_{\tau+1,2},...,x_{\tau+1,L}]$。假设 segment $s_{\tau}$第 $n$ 层的隐状态序列为 $h^n_{\tau}\in R^{L\times d}$，那么计算 segment $s_{\tau+1}$的隐状态的过程如下：

<div align=center>
    <img src="zh-cn/img/xlnet/p10.png" /> 
</div>

其中，$SG(h^{n-1}_ {\tau})$函数代表$h^{n-1}_ {\tau}$不参与梯度计算。 $[h_u。h_v]$表示向量拼接，$W^T_q,W^T_k,W^T_v$是模型参数。计算 Query 的时候用的是本段的前一层信息$h^{n-1}_ {\tau+1}$，而计算 Key 和 Value 用的是 $\tilde{h}^{n-1}_ {\tau+1}$

原则上只要 GPU 内存允许，该方法可以利用前面更多段的信息，测试阶段也可以获得更长的依赖（类似于 DenseNet）


**2.Relative Positional Encoding**

在 Transformer 中，一个重要的地方在于其考虑了序列的位置信息。在分段的情况下，如果仅仅对于每个段仍直接使用 Transformer 中的位置编码，即每个不同段在同一个位置上的表示使用相同的位置编码，就会出现问题。比如，第 $i-2$ 段和第 $i-1$ 段的第一个位置将具有相同的位置编码，但它们对于第 $i$段的建模重要性显然并不相同（例如第$i-2$段中的第一个位置重要性可能要低一些）

因此 Transformer-XL 提出了一种相对位置编码，不再关心句中词的绝对位置信息，而是相对的，比如说两个词之间隔了多少个词这样的相对信息

在标准的 Transformer 里，同一个 Segment 的 $q_i$和 $k_j$ 的 attention score 这样分解

<div align=center>
    <img src="zh-cn/img/xlnet/p11.png" /> 
</div>

其中，$E_{x_i}$是词 $i$ 的词向量，$U_i$ 是词 $i$ 的位置向量.

(a)(b)(c)(d)四项各有各的意义：(a)表示纯基于内容之间的寻址；(b)和 (c) 则分别是$i$位置的内容和位置信息分别相对于$j$位置的位置和内容信息进行的寻址；(d)则是纯基于位置之间的寻址。于是要改进的话，就需要对后三个和位置信息相关的项进行改进.

Transformer-XL 给出的改进方案是这样：

<div align=center>
    <img src="zh-cn/img/xlnet/p12.png" /> 
</div>

+ 和前面的$A^{abs}_ {i,j}$相比，第一个改动是将 (b) 和(d)里的绝对位置编码 $U_j$都替换成相对位置编码向量$R_{i-j}$。注意这里的 $R$是之前介绍的正弦函数的编码方式，它是固定的，不需要学习
+ 在 (c) 中用可训练的 $u\in R^d$替代原来的$U^T_ iW^T_ q$。因为我们假设 Attention score 只依赖于 $i$ 和 $j$ 的相对位置，而与 $i$ 的绝对位置无关，所以这里对于所有的 $i$ 都相同。也就是 $U^TW^T_ q$，所以可以用一个新的 $u$ 来表示。同理，(d)中的 $v \in R^d$  也一样
+ 最后，我们把 Key 的变换矩阵 $W_K$ 拆分成 $W_{k,E}$ 和 $W_{k,R}$，分别给内容向量和相对位置向量用

在上面的新公式里，每一项的意义都非常清晰：(a) 表示内容的计算，也就是 $x_i$ 的 Embedding 乘以变换矩阵 $W_q$ 和 $x_j$ 的 Embedding 乘以 $W_{k,E}$ 的内积；(b) 表示基于内容的位置偏置，也就是 $i$ 的向量乘以相对位置编码；(c) 表示全局的内容偏置；(d) 表示全局的位置偏置

<div align=center>
    <img src="zh-cn/img/xlnet/p13.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/xlnet/p14.png" /> 
</div>


**3.Relative Segment Encoding**

由于很多下游 NLP 任务中都包含了多个句子的情况，比如问答任务。下面我们讨论怎么在自回归框架下怎么预训练两个 segment。和 BERT 一样，我们选择两个句子，它们有 50% 的概率是连续的句子（前后语义相关），有 50% 的概率是不连续（无关) 的句子。我们把这两个句子拼接后当成一个句子来学习 Permutation LM。输入和 BERT 是类似的：[A, SEP, B, SEP, CLS]，这里 SEP 和 CLS 是特殊的两个 Token，而 A 和 B 代表两个 Segment。与 BERT 稍微不同，这里把 CLS 放到了最后。原因是因为对于 BERT 来说，Self-Attention 能够感知位置是因为我们把位置信息编码到输入向量了，Self-Attention 的计算本身不考虑位置信息。而前面我们讨论过，为了减少计算量，这里的排列语言模型通常只预测最后 1/K 个 Token。我们希望 CLS 编码所有两个 Segment 的语义，因此希望它是被预测的对象，而放到最后肯定是会被预测的

但是和 BERT 不同，XLNet 并没有增加一个预测下一个句子的 Task，原因是通过实验分析这个 Task 加进去后并不是总有帮助。[注：其实很多做法都是某些作者的经验，后面很多作者一看某个模型好，那么所有的 Follow，其实也不见得就一定好。有的时候可能只是对某个数据集有效果，或者效果好是其它因素带来的，一篇文章修改了 5 个因素，其实可能只是某一两个因素是真正带来提高的地方，其它 3 个因素可能并不有用甚至还是有少量副作用]

BERT 使用的是绝对的 Segment 编码，也就是第一个句子对于的 Segment id 是 0，而第二个句子是 1。这样如果把两个句子换一下顺序，那么输出是不一样的。XLNet 使用的是相对的 Segment 编码，它是在计算 Attention 的时候判断两个词是否属于同一个 Segment，如果位置 $i$ 和 $j$ 的词属于同一个 segment，那么使用一个可以学习的 Embedding $s_{ij}=s_{+}$，否则 $s_{ij}=s_{-}$，也就是说，我们只关心它们是属于同一个 Segment 还是属于不同的 Segment。当我们从位置 $i$ attend to $j$ 的时候，我们会这样计算一个新的 attention score：

$$a_{ij}=(q_i+b)^Ts_{ij}$$

其中 $q_i$ 是第 $i$ 个位置的 Query 向量，b 是一个可学习的 bias。最后我们会把这个 attention score 加到原来计算的 Attention score 里，这样它就能学到当 $i$ 和 $j$ 都属于某个 segment 的特征，以及 $i$ 和 $j$ 属于不同 segment 的特征


### 7.讨论与分析

**1.与BERT比较**

XLNet和BERT都是预测一个句子的部分词，但是背后的原因是不同的。BERT使用的是Mask语言模型，因此只能预测部分词(总不能把所有词都Mask了然后预测?)。而XLNet预测部分词是出于性能考虑，而BERT是随机的选择一些词来预测。

除此之外，它们最大的区别其实就是BERT是约等号，也就是条件独立的假设——那些被MASK的词在给定非MASK的词的条件下是独立的。但是我们前面分析过，这个假设并不(总是)成立。下面我们通过一个例子来说明(其实前面已经说过了，理解的读者跳过本节即可)。

假设输入是[New, York, is, a, city]，并且假设恰巧XLNet和BERT都选择使用[is, a, city]来预测New和York。同时我们假设XLNet的排列顺序为[is, a, city, New, York]。那么它们优化的目标函数分别为：

<div align=center>
    <img src="zh-cn/img/xlnet/p15.png" /> 
</div>

从上面可以发现，XLNet可以在预测York的使用利用New的信息，因此它能学到”New York”经常出现在一起而且它们出现在一起的语义和单独出现是完全不同的。


**2.XLNet的模型改进增益**

文章最后的消融分析很好地证明了乱序语言模型和 Transformer-XL 主干网络带来的提升。这部分实验采用和 BERT 一致的训练数据。以 BERT 为基础，将 BERT 的主干网络从 Transformer 换成 Transformer-XL 后，在需要建模较长上下文的阅读理解任务 RACE 和 SQuAD2.0 均有比较明显地提升 (对比 1&2 行)。而在此基础上加上乱序语言模型后，在所有任务上都有不同程度的提升 (对比 2&3 行)。 

<div align=center>
    <img src="zh-cn/img/xlnet/p16.png" /> 
</div>


**3.如何评价XLNet**

自词向量到如今以 XLNet 为代表的预训练语言模型，他们的主要区别在于对语境的不同粒度的建模：

<div align=center>
    <img src="zh-cn/img/xlnet/p17.png" /> 
</div>

XLNet 的成功来自于三点：

+ 分布式语义假设的有效性，即我们确实可以从语料的统计规律中习得常识及语言的结构。
+ 对语境更加精细的建模：从"单向"语境到"双向"语境，从"短程"依赖到"长程"依赖，XLNet 是目前对语境建模最精细的模型。
+ 在模型容量足够大时，数据量的对数和性能提升在一定范围内接近正比 [3] [4]：XLNet 使用的预训练数据量可能是公开模型里面最大的。

可以预见的是资源丰富的大厂可以闭着眼睛继续顺着第三点往前走，或许还能造出些大新闻出来，这也是深度学习给的承诺。这些大新闻的存在也渐渐堵住调参式的工作的未来，迫使研究者去思考更加底层，更加深刻的问题。

对语境的更精细建模自然是继续发展的道路，以语言模型为代表的预训练任务和下游任务之间的关系也亟待探讨。

退后一步讲，分布式语义假设的局限性在哪里？根据符号关联假设 (Symbol Interdependency Hypothesis)[5]，虽然语境的统计信息可以构建出符号之间的关系，从而确定其相对语义。但我们仍需要确定语言符号与现实世界的关系 (Language Grounding)，让我们的 AI 系统知道，「红色」对应的是红色，「天空」对应的是天空，「国家」对应的是国家。这种对应信息是通过构建知识库，还是通过和视觉、语音系统的联合建模获得？解决这一问题可能是下一大新闻的来源，也能将我们往 AI 推进一大步。

基于分布式语义假设的预训练同时受制于报道偏差 (Reporting Bias)[6]：不存在语料里的表达可能是真知识，而存在语料里面的表达也可能是假知识，更不用提普遍存在的模型偏见 (Bias) 了。我们不能因为一百个人说了「世上存在独角兽」就认为其为真，也不能因为只有一个人说了「地球绕着太阳转」便把它当做无益的噪声丢弃掉。

为了达到足够大的模型容量，我们真的需要这么大的计算量吗？已经有工作证明训练充分的 Transformer 里面存在很多重复冗余的模块 [6]。除了把网络加深加宽外，我们还有什么办法去增大模型容量的同时，保持一定的计算量？

**参考文献**


[1] Firth, J. R. (1957). Papers in linguistics 1934–1951. London: Oxford University Press.

[2] Levy O, Goldberg Y. Neural word embedding as implicit matrix factorization[C]//Advances in neural information processing systems. 2014: 2177-2185.

[3] Mahajan D, Girshick R, Ramanathan V, et al. Exploring the limits of weakly supervised pretraining[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 181-196.

[4] Hestness J, Narang S, Ardalani N, et al. Deep learning scaling is predictable, empirically[J]. arXiv preprint arXiv:1712.00409, 2017.

[5] Louwerse M M. Knowing the meaning of a word by the linguistic and perceptual company it keeps[J]. Topics in cognitive science, 2018, 10(3): 573-589.

[6] Gordon J, Van Durme B. Reporting bias and knowledge acquisition[C]//Proceedings of the 2013 workshop on Automated knowledge base construction. ACM, 2013: 25-30.

[7] Michel P, Levy O, Neubig G. Are Sixteen Heads Really Better than One?[J]. arXiv preprint arXiv:1905.10650, 2019.

