
<!-- GPT-1, GPT-2 -->
<!-- https://www.cnblogs.com/huangyc/p/9860181.html -->
<!-- https://blog.csdn.net/qq_33373858/article/details/89479038 -->
<!-- https://www.jianshu.com/p/da619f625668 -->
<!-- https://blog.csdn.net/qq_22795223/article/details/105957741?utm_medium=distribute.pc_relevant.none-task-blog-title-3&spm=1001.2101.3001.4242 -->
<!-- https://blog.csdn.net/tong_xin2010/article/details/104135852?utm_medium=distribute.pc_relevant.none-task-blog-title-4&spm=1001.2101.3001.4242 -->

<!-- https://2ly4hg.smartapps.cn/pages/article/article?articleId=336262203&authorId=129720&spm=smbd.content.share.0.1598958892579FzEpUhE&hostname=baiduboxapp&_swebfr=1 -->

<!-- https://blog.csdn.net/qq_22795223/article/details/106123065 -->
<!-- https://zhuanlan.zhihu.com/p/136138225 -->
<!-- https://zhuanlan.zhihu.com/p/96791725 -->

<!-- https://blog.csdn.net/hyzhyzhyz12345/article/details/104181606 -->

<!-- https://new.qq.com/omn/20190825/20190825A08KPF00.html -->

<!-- https://blog.csdn.net/qq_22795223/article/details/106123065 -->

<!-- GPT,1,2,3 -->
<!-- https://www.bilibili.com/video/BV1TA411Y75b?p=1 -->
<!-- https://www.bilibili.com/video/BV1At4y1D7dV?from=search&seid=11951052789676996766 -->
<!-- https://www.bilibili.com/video/BV1sz411i7fC/?spm_id_from=333.788.videocard.1 -->

## GPT-1: Improving Language Understanding by Generative Pre-Training

在GPT出现之后，通用的预训练方式是预训练整个网络然后通过fine-tune去改进具体的任务。（需要注意的是，ELMo先出现的，然后是GPT）GPT出现之后，引发了BERT，XLNet等一系列的地震式改进。对NLP任务的影响十分深远。

GPT的核心思想是**先通过无标签的文本去训练生成语言模型，再根据具体的NLP任务（如文本蕴涵、QA、文本分类等），来通过有标签的数据对模型进行fine-tuning**。与 BERT 最大的区别在于，GPT 采用了传统的语言模型进行训练，即使用单词的上文预测单词，而 BERT 是同时使用上文和下文预测单词。因此，GPT 更擅长处理自然语言生成任务 (NLG)，而 BERT 更擅长处理自然语言理解任务 (NLU)。

GPT采用的训练方法分为两步，第一步利用没有标签的文本数据集训练语言模型，第二步是根据具体的下游任务，例如 QA，文本分类等对模型进行微调，BERT 也延用了这一训练方法。我们首先了解一下GPT 与 BERT 的主要区别。

+ **预训练：**GPT 预训练的方式和传统的语言模型一样，通过上文，预测下一个单词；BERT预训练的方式是使用 Mask LM，可以同时通过上文和下文预测单词。例如给定一个句子 $[u_1, u_2, ..., u_n]$，GPT 在预测单词 $u_i$ 的时候只会利用 $[u_1, u_2, ..., u_{(i-1)}]$ 的信息，而 BERT 会同时利用 $[u_1, u_2, ..., u_{(i-1)}, u_{(i+1)}, ..., u_n]$ 的信息。如下图所示。

<div align=center>
    <img src="zh-cn/img/gpt/gpt1/p1.png" /> 
</div>

+ **模型效果：**GPT 因为采用了传统语言模型所以更加适合用于自然语言生成类的任务 (NLG)，因为这些任务通常是根据当前信息生成下一刻的信息。而 BERT 更适合用于自然语言理解任务 (NLU)。

+ **模型结构：**GPT 采用了 Transformer 的 Decoder，而 BERT 采用了 Transformer 的 Encoder。GPT 使用 Decoder 中的 Mask Multi-Head Attention 结构，在使用 $[u_1, u_2, ..., u_{(i-1)}]$ 预测单词 $u_i$ 的时候，会将 $u_i$ 之后的单词 Mask 掉。

### 1.模型结构

GPT 使用 Transformer 的 Decoder 结构，并对 Transformer Decoder 进行了一些改动，原本的 Decoder 包含了两个 Multi-Head Attention 结构，GPT 只保留了 Mask Multi-Head Attention，如下图所示。

<div align=center>
    <img src="zh-cn/img/gpt/gpt1/p2.png" /> 
</div>

GPT 使用句子序列预测下一个单词，因此要采用 Mask Multi-Head Attention 对单词的下文遮挡，防止信息泄露。例如给定一个句子包含4个单词 `[A, B, C, D]`，GPT 需要利用 `A` 预测 `B`，利用 `[A, B]` 预测 `C`，利用 `[A, B, C]` 预测 `D`。则预测 `B` 的时候，需要将 `[B, C, D]` Mask 起来。

Mask 操作是在 Self-Attention 进行 Softmax 之前进行的，具体做法是将要 Mask 的位置用一个无穷小的数替换 `-inf`，然后再 Softmax，如下图所示。

<div align=center>
    <img src="zh-cn/img/gpt/gpt1/p3.png" /> 
</div>

*GPT Mask操作*

<div align=center>
    <img src="zh-cn/img/gpt/gpt1/p4.png" /> 
</div>

*GPT Mask之后softmax*

可以看到，经过 Mask 和 Softmax 之后，当 GPT 根据单词 `A` 预测单词 `B` 时，只能使用单词 `A` 的信息，根据 `[A, B]` 预测单词 `C` 时只能使用单词 `A`, `B` 的信息。这样就可以防止信息泄露。

下图是GPT的模型结构：

<div align=center>
    <img src="zh-cn/img/gpt/gpt1/p5.png" /> 
</div>

### 2.GPT的训练过程

GPT训练过程分为两个部分，无监督预训练语言模型和有监督的下游任务fine-tuning。

**1.无监督的预训练语言模型**

给定句子$U=[u_1, u_2, ..., u_n]$，GPT 训练语言模型时需要最大化下面的似然函数。

$$L_1(U)=\sum_i\log P(u_i|u_{i-k},...,u_{i-1};\Theta)$$

这里的$k$是文本上下文窗口的大小,使用参数为$\Theta$神经网络对条件概率$P$建模，优化器为SGD（stochastic gradient descent)。 可以看到 GPT 是一个单向的模型，GPT 的输入用 $h_0$ 表示，$h_0$的计算公式如下。
$$h_0=UW_e+W_p$$
$W_p$是单词的位置的Embedding,$W_e$是单词的Embedding,$U=(u_{-k},...,u_{-1})$是上下文的词汇，用 `voc` 表示词汇表大小，`pos` 表示最长的句子长度，`dim` 表示 Embedding 维度，则 $W_p$ 是一个 `pos×dim` 的矩阵，$W_e$ 是一个 `voc×dim` 的矩阵。

得到输入$h_0$ 之后，需要将 $h_0$ 依次传入 GPT 的所有 Transformer Decoder 里，最终得到 $h_n$
$$h_l=transformerblock(h_{l-1}) \forall_{i} \in[1,n]$$
这里的$n$是模型的层数。最后得到 $h_n$ 再预测下个单词的概率。
$$P(u)=softmax(h_nW^T_{e})$$
Softmax的权重矩阵是$W^T_{e}$而输入端$W_e$也是词嵌入矩阵，这里注意一下即可。

**2.有监督的下游任务的fine-tuning**

在对模型预训练之后，采用有监督的目标任务对模型参数进行微调。假设一个有标签的数据，每一条数据为一个单词序列$x_1,x_2,...,x_m$以及相应的标签$y$,通过之前预训练的模型获得输出向量$h^m_{l}$，再sing如线性输出层，来预测标签$y$
$$P(y|x_1,...,x_m)=softmax(h_l^mW_y)$$
$W_y$ 表示预测输出时的参数，微调时候需要最大化以下函数。Loss函数为：
$$L_2(C)=\sum_{(x,y)}\log P(y|x-1,...,x_m)$$

最后，将两阶段的目标函数通过超参 $\lambda$ 相加训练整个模型：
$$L_3(C)=L_2(C)+\lambda\times L_1(C)$$

**3.具体任务的模型微调**

<div align=center>
    <img src="zh-cn/img/gpt/gpt1/p6.png" /> 
</div>

对于文本分类。只需要在预训练模型上微调。对于QA任务或文本蕴含，因为预训练模型是在连续序列上训练，只需要做一些调整，修改输入结构，经输入转化为有序序列输入。

+ **文本蕴含：**将前提$p$和假设$h$序列拼接，中间用(\$)符号来分割两个序列。
+ **文本相似度:** 分别将两个序列输入，通过模型输出两个序列的特征向量，再逐元素相加输入到线性层。
+ **问答和常识推理：**给定上下文文本$z$,问题$q$，一组可能得候选答案$\{a_k\}$,将上下文文本，问题以及每个候选答案拼接起来，得到这样一个序列$[z;q;\\$;a_k]$,再将序列输入预训练模型，经softmax层得到候选答案的概率分布。

### 3.实验

**模型细节** 我们的模型大体上和原始的transformer一致，我们训练了一个12层的只有decoder的transformer，使用有遮蔽自注意力头（包含768维状态和12个注意力头）。对于 position-wise feed-forward networks 我们使用3072维的内部状态。我们使用adam优化器，最高学习率为2.5e-4。学习率从0开始上升2000步然后通过cosine曲线下降到0，我们训练了100个epochs，64的batch size，相邻序列长度为512，因为大量使用了layernorm，我们的初始化只是用 `N(0,0.02)` 的分布。我们使用了subword（其包括wordpiece）的方式，dropout为0.1，我们同样适用了改进版L2，另外与原始transformer不同我们使用预训练的位置嵌入。

**微调细节** 我们直接使用预训练模型的参数，然后加上0.1的dropout，对于大多数任务我们的学习率为6.25e-5和32的batch大小。只需要微调3个epoch左右就能收敛。我们还使用了线性的学习率衰减，和0.2%训练步数来预热（即学习率达到达到最大值的步数）

### 4.总结

GPT 预训练时利用上文预测下一个单词，BERT 是根据上下文预测单词，因此在很多 NLU 任务上，GPT 的效果都比 BERT 要差。但是 GPT 更加适合用于文本生成的任务，因为文本生成通常都是基于当前已有的信息，生成下一个单词。

建议阅读一下 huggingface 在 Github 上的代码，里面包含了很多基于 Transformer 的模型，包括 roBERTa 和 ALBERT 等。

PT为什么不能双向？[这个答案](https://www.zhihu.com/question/322034410/answers/updated)写的非常的好，Bert、GPT、ELMo全部都提及到了。


------

## GPT-2： Language Models are Unsupervised Multitask Learners

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p66.png" /> 
</div>

<!-- https://jalammar.github.io/illustrated-gpt2/ -->
<!-- https://talktotransformer.com -->

<!-- https://2ly4hg.smartapps.cn/pages/article/article?articleId=336262203&authorId=129720&spm=smbd.content.share.0.1598958892579FzEpUhE&hostname=baiduboxapp&_swebfr=1 -->
<!-- https://www.jiqizhixin.com/articles/2019-08-26-12 -->

<!-- GPT2与GPT1的对比： https://blog.csdn.net/qq_22795223/article/details/106123065 -->
<!-- https://zhuanlan.zhihu.com/p/96791725 -->
<!-- GPT2字节对编码： https://zhuanlan.zhihu.com/p/136138225 -->

BERT、Transformer XL、XLNet 等大型自然语言处理模型轮番在各大自然语言处理任务排行榜上刷新最佳纪录，可谓你方唱罢我登场。其中，GPT-2 由于其稳定、优异的性能吸引了业界的关注

GPT-2的结构与GPT-1相比并无太大的改进均基于Transformer的Decoder结构，其在海量的数据集上训练，本部分我们将详细介绍GPT-2的模型结构。

GPT-2 有着超大的规模，它是一个在海量数据集上训练的基于 transformer 的巨大模型。GPT-2 成功的背后究竟隐藏着什么秘密？本文将带你一起探索取得优异性能的 GPT-2 模型架构，重点阐释其中关键的自注意力（self-attention）层，并且看一看 GPT-2 采用的只有解码器的 transformer 架构在语言建模之外的应用。

### 1.GPT-2和语言模型

**1.1什么是语言模型**

首先，究竟什么是语言模型（language model）？ 简单说来，语言模型的作用就是根据已有句子的一部分，来预测下一个单词会是什么。最著名的语言模型你一定见过，就是我们手机上的输入法，它可以根据当前输入的内容智能推荐下一个词。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p1.png" /> 
</div>

从这个意义上说，我们可以说 GPT-2 基本上相当于输入法的单词联想功能，但它比你手机上安装的此类应用大得多，也更加复杂。OpenAI 的研究人员使用了一个从网络上爬取的 40GB 超大数据集「WebText」训练 GPT-2，该数据集也是他们的工作成果的一部分。

如果从占用存储大小的角度进行比较，我现在用的手机输入法「SwiftKey」也就占用了 50MB 的空间，而 GPT-2 的最小版本也需要至少 500MB 的空间来存储它的全部参数，最大版本的 GPT-2 甚至需要超过 6.5GB 的存储空间。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p2.png" /> 
</div>

读者可以用「AllenAI GPT-2 Explorer」<https://gpt2.apps.allenai.org/?text=Joel%20is>或<https://talktotransformer.com>来体验 GPT-2 模型。它可以给出可能性排名前十的下一个单词及其对应概率，你可以选择其中一个单词，然后看到下一个可能单词的列表，如此往复，最终完成一篇文章。

**1.2使用 Transformers 进行语言建模**

正如本文作者在「The Illustrated Transformer 」这篇文章中所述，原始的 Transformer 模型由编码器（encoder）和解码器（decoder）组成，二者都是由被我们称为「transformer 模块」的部分堆叠而成。这种架构在机器翻译任务中取得的成功证实了它的有效性，值得一提的是，这个任务之前效果最好的方法也是基于编码器-解码器架构的。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p3.png" /> 
</div>

Transformer 的许多后续工作尝试去掉编码器或解码器，也就是只使用一套堆叠得尽可能多的 transformer 模块，然后使用海量文本、耗费大量的算力进行训练（研究者往往要投入数百甚至数千美元来训练这些语言模型，而在 AlphaStar 项目中则可能要花费数百万美元）。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p4.png" /> 
</div>

那么我们究竟能将这些模块堆叠到多深呢？事实上，这个问题的答案也就是区别不同 GPT-2 模型的主要因素之一，如下图所示。「小号」的 GPT-2 模型堆叠了 12 层，「中号」24 层，「大号」36 层，还有一个「特大号」堆叠了整整 48 层。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p5.png" /> 
</div>

**1.3与BERT的区别**

> 机器人第一法则
> 机器人不得伤害人类，或者目睹人类将遭受危险而袖手旁观。

GPT-2 是使用「transformer 解码器模块」构建的，而 BERT 则是通过「transformer 编码器」模块构建的。我们将在下一节中详述二者的区别，但这里需要指出的是，二者一个很关键的不同之处在于：GPT-2 就像传统的语言模型一样，一次只输出一个单词（token）。下面是引导训练好的模型「背诵」机器人第一法则的例子：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p6.gif" /> 
</div>

这种模型之所以效果好是因为在每个新单词产生后，该单词就被添加在之前生成的单词序列后面，这个序列会成为模型下一步的新输入。这种机制叫做自回归（auto-regression），同时也是令 RNN 模型效果拔群的重要思想。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p7.gif" /> 
</div>

GPT-2，以及一些诸如 Transformer-XL 和 XLNet 等后续出现的模型，本质上都是自回归模型，而 BERT 则不然。这就是一个权衡的问题了。虽然没有使用自回归机制，但 BERT 获得了结合单词前后的上下文信息的能力，从而取得了更好的效果。XLNet 使用了自回归，并且引入了一种能够同时兼顾前后的上下文信息的方法。

**1.4Transformer 模块的演进**

原始的 transformer 论文引入了两种类型的 transformer 模块，分别是：编码器模块和解码器模块。

1.编码器模块

首先是编码器（encoder）模块：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p8.png" /> 
</div>

原始 transformer 论文中的编码器模块可以接受长度不超过最大序列长度（如 512 个单词）的输入。如果序列长度小于该限制，我们就在其后填入预先定义的空白单词（如上图中的`<pad>`）。

2.解码器模块

其次是解码器模块，它与编码器模块在架构上有一点小差异——加入了一层使得它可以重点关注编码器输出的某一片段，也就是下图中的编码器-解码器自注意力（encoder-decoder self-attention）层。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p9.png" /> 
</div>

解码器在自注意力（self-attention）层上还有一个关键的差异：它将后面的单词掩盖掉了。但并不像 BERT 一样将它们替换成特殊定义的单词`<mask>`，而是在自注意力计算的时候屏蔽了来自当前计算位置右边所有单词的信息。

举个例子，如果我们重点关注 4 号位置单词及其前续路径，我们可以模型只允许注意当前计算的单词以及之前的单词：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p10.png" /> 
</div>

能够清楚地区分 BERT 使用的自注意力（self-attention）模块和 GPT-2 使用的带掩模的自注意力（masked self-attention）模块很重要。普通的自注意力模块允许一个位置看到它右侧单词的信息（如下左图），而带掩模的自注意力模块则不允许这么做（如下右图）。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p11.png" /> 
</div>

3.只包含解码器的模块

在 transformer 原始论文发表之后，一篇名为「Generating Wikipedia by Summarizing Long Sequences」的论文提出用另一种 transformer 模块的排列方式来进行语言建模——它直接扔掉了所有的 transformer 编码器模块……我们姑且就管它叫做「Transformer-Decoder」模型吧。这个早期的基于 transformer 的模型由 6 个 transformer 解码器模块堆叠而成：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p12.png" /> 
</div>

图中所有的解码器模块都是一样的，因此本文只展开了第一个解码器的内部结构。可以看见，它使用了带掩模的自注意力层。请注意，该模型在某个片段中可以支持最长 4000 个单词的序列，相较于 transformer 原始论文中最长 512 单词的限制有了很大的提升。

这些解码器模块和 transformer 原始论文中的解码器模块相比，除了去除了第二个自注意力层之外，并无很大不同。一个相似的架构在字符级别的语言建模中也被验证有效，它使用更深的自注意力层构建语言模型，一次预测一个字母/字符。

OpenAI 的 GPT-2 模型就用了这种只包含编码器（decoder-only）的模块。


**1.5GPT-2 内部机制速成**

> 在我内心，字字如刀；电闪雷鸣，使我疯癫。
>                              ——Budgie

接下来，我们将深入剖析 GPT-2 的内部结构，看看它是如何工作的。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p13.png" /> 
</div>

GPT-2 可以处理最长 `1024` 个单词的序列。每个单词都会和它的前续路径一起「流过」所有的解码器模块。

想要运行一个训练好的 GPT-2 模型，最简单的方法就是让它自己随机工作（从技术上说，叫做生成无条件样本）。换句话说，我们也可以给它一点提示，让它说一些关于特定主题的话（即生成交互式条件样本）。在随机情况下，我们只简单地提供一个预先定义好的起始单词（训练好的模型使用「|endoftext|」作为它的起始单词，不妨将其称为`<s>`），然后让它自己生成文字。

此时，模型的输入只有一个单词，所以只有这个单词的路径是活跃的。单词经过层层处理，最终得到一个向量。向量可以对于词汇表的每个单词计算一个概率（词汇表是模型能「说出」的所有单词，GPT-2 的词汇表中有 50000 个单词）。在本例中，我们选择概率最高的单词「The」作为下一个单词。

但有时这样会出问题——就像如果我们持续点击输入法推荐单词的第一个，它可能会陷入推荐同一个词的循环中，只有你点击第二或第三个推荐词，才能跳出这种循环。同样的，GPT-2 也有一个叫做「top-k」的参数，模型会从概率前 `k`大的单词中抽样选取下一个单词。显然，在之前的情况下，`top-k = 1`。


<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p14.gif" /> 
</div>

接下来，我们将输出的单词添加在输入序列的尾部构建新的输入序列，让模型进行下一步的预测：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p15.gif" /> 
</div>

请注意，第二个单词的路径是当前唯一活跃的路径了。GPT-2 的每一层都保留了它们对第一个单词的解释，并且将运用这些信息处理第二个单词（具体将在下面一节对自注意力机制的讲解中详述），GPT-2 不会根据第二个单词重新解释第一个单词。

**1.6更加深入了解内部原理**

1.输入编码

让我们更加深入地了解一下模型的内部细节。首先，让我们从模型的输入开始。正如我们之前讨论过的其它自然语言处理模型一样，GPT-2 同样从嵌入矩阵中查找单词对应的嵌入向量，该矩阵也是模型训练结果的一部分。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p16.png" /> 
</div>

每一行都是一个词嵌入向量：一个能够表征某个单词，并捕获其意义的数字列表。嵌入向量的长度和 GPT-2 模型的大小有关，最小的模型使用了长为 `768` 的嵌入向量来表征一个单词。

所以在一开始，我们需要在嵌入矩阵中查找起始单词`<s>`对应的嵌入向量。但在将其输入给模型之前，我们还需要引入位置编码——一些向 transformer 模块指出序列中的单词顺序的信号。1024 个输入序列位置中的每一个都对应一个位置编码，这些编码组成的矩阵也是训练模型的一部分。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p17.png" /> 
</div>

至此，输入单词在进入模型第一个 transformer 模块之前所有的处理步骤就结束了。如上文所述，训练后的 GPT-2 模型包含两个权值矩阵：嵌入矩阵和位置编码矩阵。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p18.png" /> 
</div>

将单词输入第一个 transformer 模块之前需要查到它对应的嵌入向量，再加上 `1` 号位置位置对应的位置向量。

2.堆栈之旅

第一个 transformer 模块处理单词的步骤如下：首先通过自注意力层处理，接着将其传递给神经网络层。第一个 transformer 模块处理完但此后，会将结果向量被传入堆栈中的下一个 transformer 模块，继续进行计算。每一个 transformer 模块的处理方式都是一样的，但每个模块都会维护自己的自注意力层和神经网络层中的权重。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p19.png" /> 
</div>

3.回顾自注意力机制

语言的含义是极度依赖上下文的，比如下面这个机器人第二法则:

> 机器人第二法则
> 机器人必须遵守人类给它的命令，除非该命令违背了第一法则。

我在这句话中高亮表示了三个地方，这三处单词指代的是其它单词。除非我们知道这些词指代的上下文联系起来，否则根本不可能理解或处理这些词语的意思。当模型处理这句话的时候，它必须知道：

+ 「它」指代机器人
+ 「命令」指代前半句话中人类给机器人下的命令，即「人类给它的命令」
+ 「第一法则」指机器人第一法则的完整内容

这就是自注意力机制所做的工作，它在处理每个单词（将其传入神经网络）之前，融入了模型对于用来解释某个单词的上下文的相关单词的理解。具体做法是，给序列中每一个单词都赋予一个相关度得分，之后对他们的向量表征求和。

举个例子，最上层的 transformer 模块在处理单词「it」的时候会关注「a robot」，所以「a」、「robot」、「it」这三个单词与其得分相乘加权求和后的特征向量会被送入之后的神经网络层。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p20.png" /> 
</div>

自注意力机制沿着序列中每一个单词的路径进行处理，主要由 3 个向量组成

+ 查询向量（Query 向量）：当前单词的查询向量被用来和其它单词的键向量相乘，从而得到其它词相对于当前词的注意力得分。我们只关心目前正在处理的单词的查询向量。
+ 键向量（Key 向量）：键向量就像是序列中每个单词的标签，它使我们搜索相关单词时用来匹配的对象。
+ 值向量（Value 向量）：值向量是单词真正的表征，当我们算出注意力得分后，使用值向量进行加权求和得到能代表当前位置上下文的向量。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p21.png" /> 
</div>

一个简单粗暴的比喻是在档案柜中找文件。查询向量就像一张便利贴，上面写着你正在研究的课题。键向量像是档案柜中文件夹上贴的标签。当你找到和便利贴上所写相匹配的文件夹时，拿出它，文件夹里的东西便是值向量。只不过我们最后找的并不是单一的值向量，而是很多文件夹值向量的混合。

将单词的查询向量分别乘以每个文件夹的键向量，得到各个文件夹对应的注意力得分（这里的乘指的是向量点乘，乘积会通过 softmax 函数处理）。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p22.png" /> 
</div>

我们将每个文件夹的值向量乘以其对应的注意力得分，然后求和，得到最终自注意力层的输出。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p23.png" /> 
</div>

这样将值向量加权混合得到的结果是一个向量，它将其 50% 的「注意力」放在了单词「robot」上，30% 的注意力放在了「a」上，还有 19% 的注意力放在「it」上。我们之后还会更详细地讲解自注意力机制，让我们先继续向前探索 transformer 堆栈，看看模型的输出。

4.模型输出

当最后一个 transformer 模块产生输出之后（即经过了它自注意力层和神经网络层的处理），模型会将输出的向量乘上嵌入矩阵。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p24.png" /> 
</div>

我们知道，嵌入矩阵的每一行都对应模型的词汇表中一个单词的嵌入向量。所以这个乘法操作得到的结果就是词汇表中每个单词对应的注意力得分。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p25.png" /> 
</div>

我们简单地选取得分最高的单词作为输出结果（即 `top-k = 1`）。但其实如果模型考虑其他候选单词的话，效果通常会更好。所以，一个更好的策略是对于词汇表中得分较高的一部分单词，将它们的得分作为概率从整个单词列表中进行抽样（得分越高的单词越容易被选中）。通常一个折中的方法是，将 `top-k` 设为 `40`，这样模型会考虑注意力得分排名前 `40` 位的单词。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p26.png" /> 
</div>

这样，模型就完成了一轮迭代，输出了一个单词。模型会接着不断迭代，直到生成一个完整的序列——序列达到 1024 的长度上限或序列中产生了一个终止符。

**1.7第一部分结语：大家好，这就是 GPT-2**

本部分中有一些过分简化的地方：

+ 混用了「单词」（word）和「词」（token）这两个概念。但事实上，GPT-2 使用字节对编码（Byte Pair Encoding）（我们将在侯文解释）方式来创建词汇表中的词（token），也就是说词（token）其实通常只是单词的一部分。
+ 举的例子其实是 GPT-2 在「推断/评价」（inference / evaluation）模式下运行的流程，所以一次只处理一个单词。在训练过程中，模型会在更长的文本序列上进行训练，并且一次处理多个词（token）。训练过程的批处理大小（batch size）也更大（512），而评价时的批处理大小只有 1。
+ 为了更好地组织空间中的图像，作者画图时随意转置了向量，但在实现时需要更精确。
+ Transformer 模块使用了很多归一化（normalization）层，这在训练中是很关键的。我们在「The Illustrated Transformer」<https://jalammar.github.io/illustrated-transformer/>）译文中提到了其中一些，但本文更关注自注意力层。
+ 有时文章需要用更多的小方块来代表一个向量，我把这些情况叫做「放大」，如下图所示。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p27.png" /> 
</div>

### 2.图解自注意力机制

在前面的文章中，我们用这张图来展示了自注意力机制在处理单词「it」的层中的应用：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p28.png" /> 
</div>

在本节中，我们会详细介绍该过程是如何实现的。请注意，我们将会以试图弄清单个单词被如何处理的角度来看待这个问题。这也是我们会展示许多单个向量的原因。这实际上是通过将巨型矩阵相乘来实现的。但是我想直观地看看，在单词层面上发生了什么。

**2.1自注意力机制（不使用掩模）**

首先，我们将介绍原始的自注意力机制，它是在编码器模块里计算的。先看一个简易的 transformer 模块，它一次只能处理 4 个词（token）。

自注意力机制通过以下三个主要步骤来实现：
1. 为每个路径创建查询、键和值向量。
2. 对于每个输入的词，通过使用其查询向量与其它所有键向量相乘得到注意力得分。
3. 将值向量与它们相应的注意力得分相乘后求和

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p29.png" /> 
</div>

**2.2创建查询、键和值向量**

我们重点关注第一条路径。我们用它的查询值与其它所有的键向量进行比较，这使得每个键向量都有一个对应的注意力得分。自注意力机制的第一步就是为每个词（token）路径（我们暂且忽略注意力头）计算三个向量：查询向量、键向量、值向量。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p30.png" /> 
</div>

**2.3注意力得分**

计算出上述三个向量后，我们在第二步中只用查询向量和键向量。我们重点关注第一个词，将它的查询向量与其它所有的键向量相乘，得到四个词中的每个词的注意力得分。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p31.png" /> 
</div>

**2.4求和**

现在，我们可以将注意力得分与值向量相乘。在我们对其求和后，注意力得分较高的值将在结果向量中占很大的比重。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p32.png" /> 
</div>

注意力得分越低，我们在图中显示的值向量就越透明。这是为了表明乘以一个小的数是如何削弱向量值的影响的。

如果我们在每一个路径都执行相同的操作，最终会得到一个表征每个词的向量，它包括了这个词的适当的上下文，然后将这些信息在 transformer 模块中传递给下一个子层（前馈神经网络）：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p33.png" /> 
</div>

**2.5图解掩模自注意力机制**

现在我们已经介绍了 transformer 模块中自注意力机制的步骤，接下来我们介绍掩模自注意力机制（masked self-attention）。在掩模自注意力机制中，除了第二步，其余部分与自注意力机制相同。假设模型输入只包含两个词，我们正在观察第二个词。在这种情况下，后两个词都被屏蔽了。因此模型会干扰计算注意力得分的步骤。基本上，它总是为序列中后续的词赋予 0 分的注意力得分，因此模型不会在后续单词上得到最高的注意力得分：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p34.png" /> 
</div>

我们通常使用注意力掩模矩阵来实现这种屏蔽操作。不妨想象一个由四个单词组成的序列（例如「robot must obey orders」（机器人必须服从命令））在语言建模场景中，这个序列被分成四步进行处理——每个单词一步（假设现在每个单词（word）都是一个词（token））。由于这些模型都是批量执行的，我们假设这个小型模型的批处理大小为 4，它将整个序列（包含 4 步）作为一个批处理。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p35.png" /> 
</div>

在矩阵形式中，我们通过将查询矩阵和键矩阵相乘来计算注意力得分。该过程的可视化结果如下所示，下图使用的是与单元格中该单词相关联的查询（或键）向量，而不是单词本身：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p36.png" /> 
</div>

在相乘之后，我们加上注意力掩模三角矩阵。它将我们想要屏蔽的单元格设置为负无穷或非常大的负数（例如，在 GPT2 中为 -10 亿）：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p37.png" /> 
</div>

然后，对每一行执行 softmax 操作，从而得到我们在自注意力机制中实际使用的注意力得分：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p38.png" /> 
</div>

此分数表的含义如下：

+ 当模型处理数据集中的第一个示例（第一行）时，这里只包含了一个单词（「robot」），所以 100% 的注意力都在该单词上。
+ 当模型处理数据集中的第二个示例（第二行）时，这里包含了（「robot must」），当它处理单词「must」时，48% 的注意力会在「robot」上，而另外 52% 的注意力会在「must」上。
+ 以此类推

**2.6GPT-2 的掩模自注意力机制**

接下来，我们将更详细地分析 GPT-2 的掩模自注意力机制。

1.模型评价时：一次只处理一个词

我们可以通过掩模自注意机制的方式执行 GPT-2。但是在模型评价时，当我们的模型每轮迭代后只增加一个新单词时，沿着先前已经处理过的路径再重新计算词（tokrn）的自注意力是效率极低的。

在这种情况下，我们处理第一个词（暂时忽略 `<s>`）

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p39.png" /> 
</div>

GPT-2 保存了词「a」的键向量和值向量。每个自注意力层包括了该词相应的键和值向量：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p40.png" /> 
</div>

在下一次迭代中，当模型处理单词「robot」时，它不再需要为词「a」生成查询、键和值向量。它只需要复用第一次迭代中保存的向量。

现在，在下一次迭代中，当模型处理单词 robot 时，它不再需要为 token「a」生成查询、键和值向量。它只需要复用第一次迭代中保存的向量:

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p41.png" /> 
</div>

2.GPT-2 自注意力机制:1-创建查询、键和值

假设模型正在处理单词「it」。对于下图中底部的模块来说，它对该词的输入则是「it」的嵌入向量+序列中第九个位置的位置编码：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p42.png" /> 
</div>

Transformer 中的每个模块都有自己的权重（之后会详细分析）。我们首先看到的是用于创建查询、键和值的权重矩阵。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p43.png" /> 
</div>

自注意力机制将它的输入与权重矩阵相乘（并加上一个偏置向量，这里不作图示）。

相乘后得到的向量从基本就是单词「it」的查询、键和值向量连接 的结果。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p44.png" /> 
</div>

将输入向量和注意力权重向量相乘（之后加上偏置向量）得到这个词的键、值和查询向量。

3.GPT-2 自注意力机制：1.5-分裂成注意力头

在前面的示例中，我们直接介绍了自注意力机制而忽略了「多头」的部分。现在，对这部分概念有所了解会大有用处。自注意力机制是在查询（Q）、键（K）、值（V）向量的不同部分多次进行的。「分裂」注意力头指的是，简单地将长向量重塑成矩阵形式。在小型的 GPT-2 中，有 12 个注意力头，因此这是重塑矩阵中的第一维：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p45.png" /> 
</div>

在前面的示例中，我们介绍了一个注意力头的情况。多个注意力头可以想象成这样（下图为 12 个注意力头中的 3 个的可视化结果）：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p46.png" /> 
</div>

4.GPT-2 自注意力机制：2-计算注意力得分

我们接下来介绍计算注意力得分的过程——此时我们只关注一个注意力头（其它注意力头都进行类似的操作）。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p47.png" /> 
</div>

当前关注的词（token）可以对与其它键词的键向量相乘得到注意力得分（在先前迭代中的第一个注意力头中计算得到）：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p48.png" /> 
</div>

5.GPT-2 自注意力机制：3-求和

正如前文所述，我们现在可以将每个值向量乘上它的注意力得分，然后求和，得到的是第一个注意力头的自注意力结果：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p49.png" /> 
</div>

6.GPT-2 自注意力机制：3.5-合并多个注意力头

我们处理多个注意力头的方式是先将它们连接成一个向量：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p50.png" /> 
</div>

但是这个向量还不能被传递到下一个子层。我们首先需要将这个隐含状态的混合向量转变成同质的表示形式。

7.GPT-2 自注意力机制：4-投影

我们将让模型学习如何最好地将连接好的自注意力结果映射到一个前馈神经网络可以处理的向量。下面是我们的第二个大型权重矩阵，它将注意力头的结果投影到自注意力子层的输出向量中：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p51.png" /> 
</div>

通过这个操作，我们可以生成能够传递给下一层的向量：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p52.png" /> 
</div>

8.GPT-2 全连神经网络：第一层

在全连接神经网络中，当自注意力机制已经将合适的上下文包含在其表征中之后，模块会处理它的输入词。它由两层组成：第一层的大小是模型的 4 倍（因为小型 GPT-2 的大小为 768 个单元，而这个网络将有 `768*4=3072` 个单元）。为什么是 4 倍呢？这只是原始 transformer 的运行大小（模型维度为 512 而模型的第一层为 2048）。这似乎给 transformer 模型足够的表征容量来处理目前面对的任务。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p53.gif" /> 
</div>

9.GPT-2 全连神经网络：第二层-投影到模型的维度

第二层将第一层的结果投影回模型的维度大小（小型 GPT-2 的大小为 768）。这个乘法结果是该词经过 transformer 模块处理的结果。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p54.gif" /> 
</div>


你成功处理完单词「it」了！

我们尽可能详细地介绍了 transformer 模块。现在，你已经基本掌握了 transformer 语言模型内部发生的绝大部分情况了。回顾一下，一个新的输入向量会遇到如下所示的权重矩阵：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p55.png" /> 
</div>

而且每个模块都有自己的一组权重。另一方面，这个模型只有一个词嵌入矩阵和一个位置编码矩阵：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p56.png" /> 
</div>

如果你想了解模型中的所有参数，下面是对它们的详细统计结果：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p57.png" /> 
</div>

出于某些原因，该模型共计有 1 亿 2,400 万个参数而不是 1 亿 1,700 万个。我不确定这是为什么，但是这似乎就是发布的代码中的数目（如果本文统计有误，请读者指正）。

### 3.语言建模之外

只包含解码器的 transformer 不断地表现出在语言建模之外的应用前景。在许多应用程序中，这类模型已经取得了成功，它可以用与上面类似的可视化图表来描述。在文章的最后，让我们一起来回顾一下其中的一些应用。

**机器翻译**

进行翻译时，模型不需要编码器。同样的任务可以通过一个只有解码器的 transformer 来解决：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p58.png" /> 
</div>

**自动摘要生成**

这是第一个训练只包含解码器的 transformer 的任务。也就是说，该模型被训练来阅读维基百科的文章（没有目录前的开头部分），然后生成摘要。文章实际的开头部分被用作训练数据集的标签：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p58.png" /> 
</div>

论文使用维基百科的文章对模型进行了训练，训练好的模型能够生成文章的摘要：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p60.png" /> 
</div>

**迁移学习**

在论文「Sample Efficient Text Summarization Using a Single Pre-Trained Transformer」<https://arxiv.org/abs/1905.08836>中，首先使用只包含解码器的 transformer 在语言建模任务中进行预训练，然后通过调优来完成摘要生成任务。结果表明，在数据有限的情况下，该方案比预训练好的编码器-解码器 transformer 得到了更好的效果。

GPT2 的论文也展示了对语言建模模型进行预训练后取得的摘要生成效果。

**音乐生成**

音乐 transformer<https://magenta.tensorflow.org/music-transformer>采用了只包含解码器的 transformer <https://magenta.tensorflow.org/music-transformer%EF%BC%89%E9%87%87%E7%94%A8%E4%BA%86%E5%8F%AA%E5%8C%85%E5%90%AB%E8%A7%A3%E7%A0%81%E5%99%A8%E7%9A%84transformer> 来生成具有丰富节奏和动感的音乐。和语言建模相似，「音乐建模」就是让模型以一种无监督的方式学习音乐，然后让它输出样本（我们此前称之为「随机工作」）。

你可能会好奇，在这种情境下是如何表征音乐的？请记住，语言建模可以通过对字符、单词（word）、或单词（word）某个部分的词（token）的向量表征来实现。面对一段音乐演奏（暂时以钢琴为例），我们不仅要表征这些音符，还要表征速度——衡量钢琴按键力度的指标。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p61.png" /> 
</div>

一段演奏可以被表征为一系列的 one-hot 向量。一个 MIDI 文件可以被转换成这样的格式。论文中展示了如下所示的输入序列的示例：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p62.png" /> 
</div>

这个输入序列的 one-hot 向量表征如下：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p63.png" /> 
</div>

我喜欢论文中用来展示音乐 transformer 中自注意力机制的可视化图表。我在这里加了一些注释：

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p64.png" /> 
</div>

这段作品中出现了反复出现的三角轮廓。当前的查询向量位于后面一个「高峰」，它关注前面所有高峰上的高音，一直到乐曲的开头。图中显示了一个查询向量（所有的注意力线来源）和正要处理的以前的记忆（突出了有更高 softmax 概率的音符）。注意力线的颜色对应于不同的注意力头，而宽度对应于 softmax 概率的权重。

如果你想进一步了解这种音符的表征，请观看下面的视频：<https://www.youtube.com/watch?v=ipzR9bhei_o>

**结语**

至此，我们的 GPT-2 的完全解读，以及对父类模型（只包含解码器的 transformer）的分析就到此结束了。希望读者能通过这篇文章对自注意力机制有更好的理解。在了解了 transformer 内部的工作原理之后，下次再遇到它，你将更加得心应手。感谢jalammar大神的分享:<https://jalammar.github.io/illustrated-gpt2/>


### 4.GPT-1与GPT-2的对比

GPT-2依然沿用GPT单向transformer的模式，只不过作了一些改进和改变。那GPT-2相对于GPT有哪些不同呢？

+ GPT-2去掉了fine-tuning层：不再针对于不同的任务分别进行微调建模，而是不定义这个模型应该做什么任务，模型会自动识别出来需要做什么任务，这就好比一个人博览群书，你问他什么类型的问题，他都可以顺手拈来，GPT-2就是这样一个波兰全书的模型。GPT-2的输入是完全的文本，什么提示都不加吗？当然不是，它也会加入提示词，比如：`“TL;DR:”`，GPT-2模型就会知道是做摘要工作了。输入的格式就是 `文本+TL;DR:`，然后就等待输出就行了!
+  增加数据集：既然要博览群书，当然得先有书，所以GPT-2收集了更加广泛、数量更多的语料组成数据集。该数据集包含800万个网页，大小为40G。当然这些数据集是过滤后得到的高质量文本，这样效果才能更好的哦~
+ 增加网络参数：GPT-2将Transformer堆叠的层数增加到48层，隐层的维度为1600，参数量更是达到了15亿。15亿什么概念呢，Bert的参数量也才只有3亿,当然，这样的参数量也不是说谁都能达到的，这也得取决于money的多少.
+ 调整transformer：将layer normalization放到每个sub-block之前，并在最后一个Self-attention后再增加一个layer normalization。论文中这块感觉说的模棱两可，如果给个图就好了。不过可以通过代码了解这一细节，下图是我理解如何加layer normalization的示意图，给大家做个参考

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p65.png" /> 
</div>

+ 其他：GPT-2将词汇表数量增加到50257个；最大的上下文大小 (context size) 从GPT的512提升到了1024 tokens；batchsize增加到512。

### 5.BPE（Byte Pair Encoding）

在模型输入方面，GPT-2 采用的是 Byte Pair Encoding(以下简称 BPE)的 Subword 算法。BPE 是一种简单的数据压缩形式，可以有效地平衡词汇表大小和编码所需的 token 数量。它可以提高词表的空间使用效率，避免得到类似 ‘dog.’、‘dog!’、‘dog?’ 的词。
BPE 和我们之前的提到的 WordPiece 的区别在于，WordPiece 是基于概率生成 Subword 的，而 BPE 是基于贪心策略，每次都取最高频的字节对。

**常见的文本切分粒度**

我们在切分文本的时候，使用的粒度是字、词、ngram。而我们说的“嵌入”，指的是将字符、词语或者网络节点等具有现实意义的元素，用稠密向量表示为语义空间中的一个点。

对中文和英文来说，字符级别的嵌入都有一定的困难：英文中有意义的文本会比较长(输入序列对应的时间步较多)，训练会比较慢，遗忘也会比较严重；中文有几万个字(分类标签较多)，训练也比较慢。词嵌入的困难就更明显了，当前主要语言的词汇表都是十万级别大小，导致模型训练非常耗时。

为了控制词汇表规模，大家做了很多尝试:去除停用词；对英文文本做tokenization；将“不重要的”字、词替换为统一的符号；将数字替换为统一的符号；(中文中)将英文字符替换为统一的符号；字嵌入；等等。这些方法，以及它们的组合，效果各有千秋。删掉一些token损失信息；而保留生僻token的话，多多少少又会(因为没有充分学习)对模型的效果产生负面影响。

Sennrich, R等人(2015)搞了一次尝试，使用介于字符和词之间的一种文本粒度对文本进行切分[Sennrich, R., Haddow, B., and Birch, A. Neural machine translation of rare words with subword units]，以进一步控制词汇表规模，并缓解数据稀疏问题。这种切分方式有点类似ngram，不是字符，也不是词语，不适合人类阅读，但是对机器非常亲和：词汇量被控制在一个不大不小的规模。他们采用的方法，叫做字节对编码。

**一种奇怪的文本切分粒度——字节对编码**

Gage, Philip(1994)提出了一种数据压缩算法，叫做字节对编码(Byte Pair Encoding, BPE)。这个算法的大意是这样的:用一个新代号表示数据中最常见的bigram(可以是字节对、字符对、词语对等等)，不断迭代，直到剩余bigram的频率为1(或者可以自定义一个终止条件)。其操作方式很简单，这里以压缩字符串“aabcaacc”为例,如下图所示。

<div align=center>
    <img src="zh-cn/img/gpt/gpt2/p67.png" /> 
</div>


**Byte-level字节对编码**

前面介绍的BPE，是在字符的基础上进行压缩，应该叫“二元字符编码”。而GPT-2剑走偏锋，选择了一种对机器更加亲和的粒度，即在字节的基础上、使用BPE算法对文本进行压缩——得到的subword序列，就是文本的新型表示。下游的“嵌入”，也是在subword的基础上进行嵌入。

字节级BPE的细节，可以参考文献[<https://arxiv.org/pdf/1909.03341.pdf>]。字节级别的BPE，你懂的，模式非常少(当然比英文字母的个数要多)，不仅控制了词汇表规模，还极大的缓解了数据稀疏问题，因此可以支撑更好的分布式表示。

个人以为，字节级别的BPE是GPT-2能力的重要来源。

------

## GPT-3: Language Models are Few-Shot Learners


<!-- https://blog.csdn.net/weixin_42137700/article/details/107893052 -->
<!-- http://www.360doc.com/content/20/0804/08/7673502_928418449.shtml -->
<!-- https://blog.csdn.net/weixin_42137700/article/details/107860376 -->

<!-- https://www.bilibili.com/video/BV1TA411Y75b?p=2 -->
<!-- https://www.bilibili.com/video/BV1FA411B7Sp?from=search&seid=3700057539523989190 -->
<!-- https://www.bilibili.com/video/BV1At4y1D7dV?from=search&seid=3700057539523989190 -->

<!-- https://mp.weixin.qq.com/s/ZuJipGApFAsFJtBNSgBcRg -->
<!-- https://huggingface.co/transformers/tokenizer_summary.html -->
<!-- https://blog.csdn.net/weixin_41089007/article/details/106501248 -->
<!-- https://zhuanlan.zhihu.com/p/174127926?utm_source=wechat_session&utm_medium=social&utm_oi=873108524698308608&utm_campaign=shareopn -->
<!-- https://zhuanlan.zhihu.com/p/210392010?utm_source=wechat_session&utm_medium=social&utm_oi=873108524698308608&utm_campaign=shareopn -->

<!-- https://zhuanlan.zhihu.com/p/148488261?utm_source=wechat_session&utm_medium=social&utm_oi=873108524698308608&utm_campaign=shareopn -->
<!-- https://mbd.baidu.com/newspage/data/landingshare?pageType=1&isBdboxFrom=1&uk=dOycTSIJne3oxxhcIEX-NQ&context=%7B%22nid%22%3A%22news_9817526512547947949%22%7D -->

<video id="video" controls="" preload="none" poster="http://om2bks7xs.bkt.clouddn.com/2017-08-26-Markdown-Advance-Video.jpg" width="1000">
<source id="mp4" src="zh-cn/img/gpt/gpt3/GPT-3.mp4" type="video/mp4">
</video>

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p0.png" /> 
</div>

### 1.模型结构和参数量

**模型结构：**

该研究使用了和 GPT-2 相同的模型和架构，包括改进的初始设置、预归一化和 reversible tokenization。区别在于 GPT-3 在 transformer 的各层上都使用了交替密集和局部带状稀疏的注意力模式，类似于 Sparse Transformer [CGRS19]（感兴趣可以自行研究）。

为了研究性能对模型大小的依赖性，该研究训练了 8 种不同的模型大小，涵盖 3 个数量级，从 1.25 亿参数到 1750 亿个参数不等，具备 1750 亿个参数的模型即为 GPT-3。

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p1.png" /> 
</div>

上图表展示了 8 个模型的大小和架构。这里 $n_{params}$表示可训练参数总量，$n_{layers}$ 表示层数，$d_{model}$表示每个瓶颈层中的单元数量（在该研究中前馈层总是瓶颈层大小的 4 倍，即 $d_{ff} = 4 d_{model}$），$d_{head}$ 表示每个注意力头的维度。所有的模型均使用 $n_{ctx} = 2048 tokens$的语境窗口。

**参数量:**

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p2.png" /> 
</div>

和往常一样，GPT-3 立即放出了 GitHub 项目页面，不过目前仅是一些生成样本和数据集，还没有代码：<https://github.com/openai/gpt-3>。

不过上传的没有那么快其实情有可原，在 issue 里有人道出了真相：参数这么多，如果按照 GPT-2 15亿参数等于 6G 这么算的话，GPT-3 模型可能要 700G，老硬盘还装不下，个人和没有实力的企业就不要在考虑使用这个模型了！！！

**训练成本:**

+ 有一种说法是训练GPT-3花费了460万美元:

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p3.gif" /> 
</div>

有人对GPT的未来进行了描绘。「GPT-4 将会比人类写得更好，而且，当它对自己的答案不确定时，还可以进行研究。」GPT-4作为进阶版，将更有钻研精神。

此前Open AI的论文中曾经提到，自2012年以来，要训练一个人工智能模型在基准测试ImageNet图像分类任务中达到同等的分类效果，所需的算力每16个月就会减少1/2。算法演进速度吊打摩尔定律。

也就是说，在过去的7年内，训练神经网络的效率每16个月就会翻一番。照这个速度，新一代的GPT或许会cheap&good！

MIT研究员Lex Fridman预测，训练GPT-4预计会花费26亿美元。2024年，训练GPT-4只需要花费4000万美元，到2032年，估计只需要花费500万美元。

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p4.png" /> 
</div>

+ 另一种说法是训练GPT-3需要1200万美元

你肯定想问这样一个问题：训练 GPT-3 模型需要花多少钱？我们目前还只能粗略地估计——训练一个 BERT 模型租用云算力要花大概 6912 美元，训练 GPT-2 每小时要花费 256 美元，但 OpenAI 一直没有透露一共要花多少小时。

相比之下，GPT-3 需要的算力（flops）是 BERT 的 1900 多倍，所以这个数字应该是千万美元级别的，以至于研究者在论文第九页说：我们发现了一个 bug，但没钱再去重新训练模型，所以先就这么算了吧。


### 2.GPT-3的训练数据

GPT-3使用了**45TB**数据进行训练

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p5.png" /> 
</div>


### 3.GPT-3的核心创新点

训练方法，去掉了Fine-Tuning

+ 传统方法：Fine-Tuning
+ GPT-3使用方法：

OpenAI 团队使用的基础预训练方法包括模型、数据与训练三部分。GPT-3 的训练过程与 GPT-2 类似，但对模型大小、数据集大小与多样性、训练长度都进行了相对直接的扩充。关于语境学习，GPT-3 同样使用了与 GPT-2 类似的方法，不过 GPT-3 研究团队系统地探索了不同的语境学习设定。

OpenAI 团队明确地定义了用于评估 GPT-3 的不同设定，包括 zero-shot、one-shot 和 few-shot。

**Few-Shot（FS）：**指的是在推理时对模型进行一些任务相关的示例演示，但不允许权重更新。如图2.1所示，对于一个典型的数据集，一个示例具有上下文和所需的补全（例如英语句子和对应的法语句子），并通过给出K个示例上下文和补全的例子进行了Few-Shot。我们通常将K设置在10到100的范围内。FS的主要优点是，大大减少了对特定任务数据的需求，并减少了过拟合的可能性。主要缺点是，到目前为止，这种方法的结果要比最新的微调模型差很多。而且，仍然需要少量的任务特定数据。

**One-Shot(1S)：**和FS一样，不允许权重更新，但是k设置为1，和人类处理任务最为相似。

**Zero-Shot (0S) ：**没有示例演示，仅向模型提供描述任务的自然语言指令，同样没有权重更新。

四种方法对比见下图:

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p6.png" /> 
</div>

上图以英-法翻译任务为例，展示了四种方法。该研究将重点放在 zero-shot、one-shot 和 few-shot 上，其目的并非将它们作为竞品进行比较，而是作为不同的问题设置。OpenAI 团队特别强调了 few-shot 结果，因为其中许多结果仅仅略微逊色于 SOTA 微调模型。不过，用 one-shot 甚至有时是 zero-shot 与人类水平进行对比似乎最为公平，这也是未来工作的重要目标之一。

### 4.部分有意思的测试结果

关于测试结果，详细的可以参考GPT-3的原始论文，这里仅展示部分有意思的此时结果。

**新闻生成**

据《华盛顿邮报》报道，经过两天的激烈辩论，联合卫理公会同意了一次历史性的分裂：要么创立新教派，要么则在神学和社会意义上走向保守。大部分参加五月份教会年度会议的代表投票赞成加强任命 LGBTQ 神职人员的禁令，并制定新的规则「惩戒」主持同性婚礼的神职人员。但是反对这些措施的人有一个新计划：2020 年他们将形成一个新教派「基督教卫理公会」。

《华盛顿邮报》指出，联合卫理公会是一个自称拥有 1250 万会员的组织，在 20 世纪初期是「美国最大的新教教派」，但是近几十年来它一直在萎缩。这次新的分裂将是该教会历史上的第二次分裂。第一次发生在 1968 年，当时大概只剩下 10% 的成员组成了「福音联合弟兄会」。《华盛顿邮报》指出，目前提出的分裂「对于多年来成员不断流失的联合卫理公会而言，来得正是时候」，这「在 LGBTQ 角色问题上将该教派推向了分裂边缘」。同性婚姻并不是分裂该教会的唯一问题。2016 年，该教派因跨性别神职人员的任命而分裂。北太平洋地区会议投票禁止他们担任神职人员，而南太平洋地区会议投票允许他们担任神职人员。

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p7.png" /> 
</div>

这确定不是报刊记者撰写的短新闻吗？

给出标题「联合卫理公会同意这一历史性分裂」和子标题「反对同性恋婚姻的人将创建自己的教派」，GPT-3 生成了上述新闻。

在 OpenAI 的测试中，人类评估人员也很难判断出这篇新闻的真假，检测准确率仅为 12%。

**GPT-3 的造句能力**

给出一个新单词及其定义，造出一个新句子。难吗？这需要你理解单词的意义及适用语境。OpenAI 研究者测试了 GPT-3 在这一任务上的能力：给出一个不存在的单词（如「Gigamuru」），令 GPT-3 使用它造句

我们来看 GPT-3 的生成结果：

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p8.png" /> 
</div>

给出新单词「Gigamuru」（表示一种日本乐器）。GPT-3 给出的句子是：叔叔送了我一把 Gigamuru，我喜欢在家弹奏它。严丝合缝，非常合理，完美！

**语法纠错**

给出一句带有语法错误的话，让 GPT-3 进行修改。

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p10.png" /> 
</div>

第一个例子中，原句里有两个并列的动词「was」和「died」，GPT-3 删除系动词「was」，将其修改为正确的句子。

**GPT-3 还能做计算题？**

penAI 研究人员在以下 10 项任务中测试了 GPT-3 做简单计算的能力，且无需任何任务特定的训练。

这十项任务分别是：两位数加减法、三位数加减法、四位数加减法、五位数加减法、两位数乘法，以及一位数混合运算。

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p9.png" /> 
</div>

用于测试 GPT-3 计算能力的十项任务。

在这十项任务中，模型必须生成正确的答案。对于每项任务，该研究生成包含 2000 个随机实例的数据集，并在这些实例上评估所有模型。

下图展示了 GPT-3（few-shot）在这十项计算任务上的性能。从图中可以看到，小模型的性能较差，即使是拥有 130 亿参数的模型（仅次于拥有 1750 亿的 GPT-3 完整版模型）处理二位数加减法的准确率也只有 50% 左右，处理其他运算的准确率还不到 10%。

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p11.png" /> 
</div>

**GPT-3生成articles**

<div align=center>
    <img src="zh-cn/img/gpt/gpt3/p12.png" /> 
</div>


还有很多很多GPT-3的实验，可以参考原论文！

当然也有很多人对GPT-3的识别效果产生了质疑：[马库斯开喷GPT-3：演员而已，它根本不知道自己在说什么](https://zhuanlan.zhihu.com/p/210392010?utm_source=wechat_session&utm_medium=social&utm_oi=873108524698308608&utm_campaign=shareopn)

GPT-3的模型和代码并没有开源，即使开源我们也无法使用！！！

