## MASS: Masked Sequence to Sequence Pre-training for Language Generation

<!-- date 2020-09-19 -->
<!-- https://zhuanlan.zhihu.com/p/191408017 -->
<!-- https://zhuanlan.zhihu.com/p/67687640 -->
<!-- https://zhuanlan.zhihu.com/p/67891175 -->
<!-- https://blog.csdn.net/qq_42793029/article/details/93851233 -->
<!-- https://www.microsoft.com/en-us/research/blog/introducing-mass-a-pre-training-method-that-outperforms-bert-and-gpt-in-sequence-to-sequence-language-generation-tasks/ -->

<!-- https://www.microsoft.com/en-us/research/publication/mass-masked-sequence-to-sequence-pre-training-for-language-generation/ -->
<!-- https://github.com/microsoft/MASS -->


### 1.引言

微软亚洲研究院的研究员在ICML 2019上提出了一个全新的通用预训练方法MASS, 在序列到序列的自然语言生成任务中全面超越BERT和GPT。 BERT在自然语言理解（比如情感分类、自然语言推理、命名实体识别、SQuAD阅读理解等）任务中取得了很好的结果，受到了越来越多的关注。然而，在自然语言处理领域，除了自然语言理解任务，还有很多**序列到序列**的自然语言生成任务，比如机器翻译、文本摘要生成、对话生成、问答、文本风格转换等。在这类任务中，目前主流的方法是`编码器-注意力-解码器`框架。

编码器（Encoder）将源序列文本`X`编码成隐藏向量序列，然后解码器（Decoder）通过注意力机制（Attention）抽取编码的隐藏向量序列信息，自回归地生成目标序列文本`Y`。

<div align=center>
    <img src="zh-cn/img/mass/p1.png" /> 
</div>

*编码器-注意力-解码器框架*

BERT 和 XLnet 通常是对一个编码器进行自然语言理解的预训练；而 GPT 则是对一个解码器进行语言建模的预训练。当利用 BERT 和 GPT 进行序列到序列的语言生成任务时，我们通常需要对编码器和解码器分别进行预训练。在这种情况下，`编码器 - 注意力 - 解码器`框架和注意力机制并没有得到联合训练。然而，注意力机制在这类任务中极为重要，一旦缺失便会导致 BERT 和 GPT 无法达到最佳性能。


### 2.新的预训练方法——MASS

针对序列到序列的自然语言生成任务，微软亚洲研究院的机器学习小组提出了一种新的预训练方法，即Mask的序列到序列预训练（MASS：Masked Sequence to Sequence Pre-Training）。MASS 随机Mask一个长度为 `k` 的句子片段，并通过`编码器 - 注意力 - 解码器`框架预测这一被掩蔽的片段。

<div align=center>
    <img src="zh-cn/img/mass/p2.png" /> 
</div>

*MASS 框架*

如上图所示，编码器端的第`3-6个`词被屏蔽掉，然后解码器端只预测这几个连续的词，而屏蔽掉其它词，图中`_`代表被屏蔽的词。

MASS 预训练具有以下优势：

+ 解码器端的其他标记（在编码器端未被掩蔽的标记）被掩蔽，从而推动解码器提取更多信息以帮助预测连续句子片段，促进编码器-注意力-解码器结构的联合训练；
+ 为了给解码器提供更多有用的信息，编码器被强制提取未被掩蔽的标记的含义，这可以提高编码器理解源序列文本的能力；
+ 解码器被设计用以预测连续的标记（句子片段），这可以提升解码器的语言建模能力。

### 3.统一的预训练框架

MASS 有一个重要的超参数 `k`（被掩蔽的片段的长度）。通过调整 `k` 值，MASS 可以将 BERT 中掩蔽的语言建模和 GPT 中的标准语言建模结合起来，从而将 MASS 扩展成一个通用的预训练框架。

当 `k=1` 时，根据 MASS 的设计，编码器端的一个标记被掩蔽，而解码器端则会预测出该掩蔽的标记，如下图所示。解码器端没有输入信息，因而 MASS 等同于 BERT 中掩蔽的语言模型。

<div align=center>
    <img src="zh-cn/img/mass/p3.png" /> 
</div>

*k=1时，编码器端一个标记被掩蔽，而解码器端则会预测出该掩蔽的标记*

当`k=m`（m 是序列的长度）时，在 MASS 中，编码器端的所有标记都被掩蔽，而解码器端会预测所有的标记，如下图所示。解码器端无法从编码器端提取任何信息，MASS 等同于 GPT 中的标准语言模型。

<div align=center>
    <img src="zh-cn/img/mass/p4.png" /> 
</div>

*k=m时，编码器端的所有词都被掩蔽，而解码器端会预测所有的标记，等同于GPT中的标准语言模型*


不同·
`k`值下 MASS 的概率公式如下表所示，其中 `m` 是序列的长度，`u` 和 `v` 分别是Mask片段的起始和终止位置，$X^{u:v}$代表从位置 `u` 到 `v`的标记都被掩蔽的序列。可以看出，当 `k=1` 或 `m` 时，MASS 的概率公式等同于 BERT 中的被掩蔽的语言模型和 GPT 中的标准语言模型。

MASS预训练的loss 函数如下(以极大似然函数作为目标函数):

<div align=center>
    <img src="zh-cn/img/mass/p5.png" /> 
</div>

注: $x^{u:v}$为以句子位置 `u`为起点，`v`为终点；  $x^{\u:v}$为以句子位置`u`为起点，`v`为终点之外的部分

<div align=center>
    <img src="zh-cn/img/mass/p6.png" /> 
</div>

*在不同k值下MASS的概率公式*

研究人员通过实验来分析了在不同`k`值下的 MASS 性能，如下图所示：

<div align=center>
    <img src="zh-cn/img/mass/p7.png" /> 
</div>

*在训练前和微调阶段的各种掩蔽长度 k 下 MASS 的表现，其中包括 a) 英语句子预训练模型的PPL b) WMT13 英语-法语翻译的法语句子 c) WMT13 无监督英语-法语翻译的 BLEU 值 d) 文本摘要生成的 ROUGE 值 e) 对话生成的PPL*

当`k` 等于句子长度的一半时，下游任务可以达到其最佳性能。掩蔽句子中一半的词可以很好地平衡编码器和解码器的预训练部分。如果预训练更偏向编码器端（`k=1`，即 BERT）或更偏向解码器端（`k=m`，LM / GPT），则无法实现最优的性能，这也表现出了 MASS 在序列到序列的语言生成任务中的优势。


### 4.序列到序列的语言生成任务测试

+ 预训练

值得注意的是，MASS 仅需要无监督的单语数据进行预训练（例如 WMT News Crawl Data、Wikipedia Data 等）。MASS 支持跨语言任务（例如机器翻译）和单语任务（例如文本摘要生成、对话生成）。在对英语-法语翻译等跨语言任务进行预训练时，研究人员可以在一个模型中同时进行英语-英语和法语-法语的预训练，并使用附加的语言嵌入向量来区分语言。在无监督的机器翻译、文本摘要生成和对话生成四个领域，研究人员对 MASS 进行了微调，以验证其有效性。

+ 无监督机器翻译

关于无监督机器翻译任务，研究人员将 MASS 与之前的方法进行了比较，包括以前最先进的方法 Facebook XLM。XLM 使用了由 BERT 创建的掩蔽预训练语言模型，以及标准语言模型来分别预训练编码器和解码器。

结果如下表所示，MASS 在 WMT14 英语-法语、WMT16 英语-德语和英语-罗马尼亚语的六个翻译方向上的表现都优于 XLM，并取得了最新的最优结果。

<div align=center>
    <img src="zh-cn/img/mass/p8.png" /> 
</div>

* MASS 与之前关于无监督机器翻译方法之间的比较；英语-法语翻译报道在 newstest2014 上，其它的在 newstest2016 可以找到；由于 XLM 在编码器和解码器中使用 MLM 和 CLM 的不同组合，因此报告上显示的是每个语言对上 XLM 的最高 BLEU 值*


+ 低资源机器翻译

低资源机器翻译是指使用有限的双语训练数据来进行机器翻译。研究人员模拟了 WMT14 英语-法语，WMT16 英语-德语和英语-罗马尼亚语翻译（分别为 10K，100K 和 1M 双语数据）的低资源情景。

<div align=center>
    <img src="zh-cn/img/mass/p9.png" /> 
</div>

*MASS 与低资源机器翻译方法之间的比较*

上图显示MASS在不同数据规模上的表现，均比不用预训练的基线模型有不同程度的提升，并随着监督数据越少，提升效果越显著。

+ 文本摘要生成

研究人员将 MASS 与 BERT+LM（编码器用 BERT 预训练，解码器用标准语言模型 LM 预训练）、DAE（去噪自编码器）进行了比较。从下表中可以看出，MASS 的表现都优于 BERT+LM 和 DAE。


<div align=center>
    <img src="zh-cn/img/mass/p10.png" /> 
</div>

*文本摘要生成任务中，MASS 和两种预训练方法之间的比较*

+ 对话生成

研究人员将 MASS 和 BERT+LM 进行了比较。下表显示 MASS 实现了比 BERT+LM 更低的 PPL。

<div align=center>
    <img src="zh-cn/img/mass/p11.png" /> 
</div>

*MASS 与 BERT+LM 之间的比较数据*

MASS 连续在序列到序列的语言生成任务上实现显著增益，Facebook 的研究者表示，期待今后在自然语言理解任务中测试 MASS 的性能，并希望在未来的工作中，将 MASS 的应用领域扩展到包含语音、视频等其它序列到序列的生成任务中。

### 5.实验细节

预训练

+ 随机替换连续的token,并用[M]替换，起始位置随机设为u
+ 与bert论文一样，80%用来mask， 10%随机替换，10%保持不变
+ Mask的长度设为句子长度的0.5
+ 为了减少内存和时间消耗，在decoder部分移除了padding,但保留了positional embedding, 这样就可以减少decoder部分50% 计算量
+ 语言：因为要应用到机器翻译任务，所以预训练模型采用4种语言，作者把四种语言同时进行Byte-Pair Encoding，生成一个60k的词表。在预训练的时候也是同时用四种语言的语料，即一个batch是`32X4`个句子。
+ 预训练LR=1e-4，NMT的精调LR=1e-4。


Fine-Tuning on Text Summarization

Fintune阶段以article作为decoder输入，title作为encoder输入。最终用ROUGE-1,ROUGE-2,ROUGE-L作为评估指标。Inference时beam size设为5


### 6.结论

+ MASS达到了机器翻译的新SOTA
+ MASS > BERT+LM式的预训练
+ Mask掉连续的token，可以得到更好的语言建模能力（已实验验证）
+ 只让decoder从encoder侧获取信息，得到的模型效果更好（已实验验证）

总体来讲，MASS还是给我开阔了新的思路，其实仔细想这个想法也是不难想出来的，关键还是要动手去验证，并且花心思去提升效果，细节见英雄。