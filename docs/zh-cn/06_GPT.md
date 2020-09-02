
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

<!-- https://jalammar.github.io/illustrated-gpt2/ -->
<!-- https://talktotransformer.com -->





------

## GPT-3: Language Models are Few-Shot Learners