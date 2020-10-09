## BERT-WWM: Pre-Training with Whole Word Masking for Chinese BERT

<!-- https://www.cnblogs.com/huangyc/p/10223075.html -->

<!-- https://zhuanlan.zhihu.com/p/103177424 -->

<!-- https://www.cnblogs.com/dyl222/p/11845126.html -->

<!-- https://github.com/ymcui/Chinese-BERT-wwm -->

<!-- https://www.jianshu.com/p/1008921d6a2b -->

<!-- https://blog.csdn.net/weixin_37947156/article/details/99235621 -->
<!-- https://zhuanlan.zhihu.com/p/75987226 -->


<!-- https://mp.weixin.qq.com/s/2nOiF5HCzFkGGNp53aldPA -->

<!-- https://mp.weixin.qq.com/s?__biz=MzU2NDQ3MTQ0MA==&mid=2247484698&idx=1&sn=bd4e7511c3bde1ffd6eea149c50f782a&chksm=fc4b36e5cb3cbff357008627bd2da03d341d30295e3ab891131deae32e56423101e609c22258&scene=21#wechat_redirect -->

### 1.WordPiece原理

<!-- https://www.cnblogs.com/huangyc/p/10223075.html -->

现在基本性能好一些的NLP模型，例如OpenAI GPT，google的BERT，在数据预处理的时候都会有WordPiece的过程。WordPiece字面理解是把word拆成piece一片一片，其实就是这个意思。

WordPiece的一种主要的实现方式叫做BPE（Byte-Pair Encoding）双字节编码。

BPE的过程可以理解为把一个单词再拆分，使得我们的词表会变得精简，并且寓意更加清晰。比如"loved","loving","loves"这三个单词。其实本身的语义都是“爱”的意思，但是如果我们以单词为单位，那它们就算不一样的词，在英语中不同后缀的词非常的多，就会使得词表变的很大，训练速度变慢，训练的效果也不是太好。

BPE算法通过训练，能够把上面的3个单词拆分成"lov","ed","ing","es"几部分，这样可以把词的本身的意思和时态分开，有效的减少了词表的数量。

**BPE算法**

BPE的大概训练过程：首先将词分成一个一个的字符，然后在词的范围内统计字符对出现的次数，每次将次数最多的字符对保存起来，直到循环次数结束。

我们模拟一下BPE算法。

我们原始词表如下：
```
{'l o w e r </w>': 2, 'n e w e s t </w> ': 6, 'w i d e s t </w> ': 3, 'l o w </w>': 5}
```

其中的key是词表的单词拆分层字母，再加`</w>`代表结尾，value代表词出现的频率。下面我们每一步在整张词表中找出频率最高相邻序列，并把它合并，依次循环。

```
原始词表 {'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3, 'l o w </w>': 5}
出现最频繁的序列 ('s', 't') 9
合并最频繁的序列后的词表 {'n e w e st </w>': 6, 'l o w e r </w>': 2, 'w i d e st </w>': 3, 'l o w </w>': 5}
出现最频繁的序列 ('e', 'st') 9  # e后结st的频率
合并最频繁的序列后的词表 {'l o w e r </w>': 2, 'l o w </w>': 5, 'w i d est </w>': 3, 'n e w est </w>': 6}
出现最频繁的序列 ('est', '</w>') 9
合并最频繁的序列后的词表 {'w i d est</w>': 3, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'l o w </w>': 5}
出现最频繁的序列 ('l', 'o') 7
合并最频繁的序列后的词表 {'w i d est</w>': 3, 'lo w e r </w>': 2, 'n e w est</w>': 6, 'lo w </w>': 5}
出现最频繁的序列 ('lo', 'w') 7
合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'n e w est</w>': 6, 'low </w>': 5}
出现最频繁的序列 ('n', 'e') 6
合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'ne w est</w>': 6, 'low </w>': 5}
出现最频繁的序列 ('w', 'est</w>') 6
合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'ne west</w>': 6, 'low </w>': 5}
出现最频繁的序列 ('ne', 'west</w>') 6
合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'newest</w>': 6, 'low </w>': 5}
出现最频繁的序列 ('low', '</w>') 5
合并最频繁的序列后的词表 {'w i d est</w>': 3, 'low e r </w>': 2, 'newest</w>': 6, 'low</w>': 5}
出现最频繁的序列 ('i', 'd') 3
合并最频繁的序列后的词表 {'w id est</w>': 3, 'newest</w>': 6, 'low</w>': 5, 'low e r </w>': 2}
```
这样我们通过BPE得到了更加合适的词表了，这个词表可能会出现一些不是单词的组合，但是这个本身是有意义的一种形式，加速NLP的学习，提升不同词之间的语义的区分度。

### 2.摘要

<!-- https://mp.weixin.qq.com/s?__biz=MzU2NDQ3MTQ0MA==&mid=2247484698&idx=1&sn=bd4e7511c3bde1ffd6eea149c50f782a&chksm=fc4b36e5cb3cbff357008627bd2da03d341d30295e3ab891131deae32e56423101e609c22258&scene=21#wechat_redirect -->

基于Transformers的双向编码表示（BERT）在多个自然语言处理任务中取得了广泛的性能提升。近期，谷歌发布了基于全词覆盖（Whold Word Masking）的BERT预训练模型，并且在SQuAD数据中取得了更好的结果。应用该技术后，在预训练阶段，同属同一个词的WordPiece会被全部覆盖掉，而不是孤立的覆盖其中的某些WordPiece，进一步提升了Masked Language Model （MLM）的难度。在本文中我们将WWM技术应用在了中文BERT中。我们采用中文维基百科数据进行了预训练。该模型在多个自然语言处理任务中得到了测试和验证，囊括了句子级到篇章级任务，包括：情感分类，命名实体识别，句对分类，篇章分类，机器阅读理解。实验结果表明，基于全词覆盖的中文BERT能够带来进一步性能提升。同时我们对现有的中文预训练模型BERT，ERNIE和本文的BERT-wwm进行了对比，并给出了若干使用建议。预训练模型将发布在：<https://github.com/ymcui/Chinese-BERT-wwm>

### 3.简介

Whole Word Masking (WWM)，暂翻译为全词Mask，是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个词缀，在生成训练样本时，这些被分开的词缀会随机被[MASK]替换。在全词Mask中，如果一个完整的词的部分WordPiece被[MASK]替换，则同属该词的其他部分也会被[MASK]替换，即全词Mask。

同理，由于谷歌官方发布的BERT-base（Chinese）中，中文是以字为粒度进行切分，没有考虑到传统NLP中的中文分词（CWS）。我们将全词Mask的方法应用在了中文中，即对组成同一个词的汉字全部进行[MASK]。该模型使用了中文维基百科（包括简体和繁体）进行训练，并且使用了哈工大语言技术平台LTP（<http://ltp.ai>）作为分词工具。

下述文本展示了全词Mask的生成样例。

<div align=center>
    <img src="zh-cn/img/bert-wwm/p1.png" /> 
</div>


### 4.基线测试结果

我们选择了若干中文自然语言处理数据集来测试和验证预训练模型的效果。同时，我们也对近期发布的谷歌BERT，百度ERNIE进行了基准测试。为了进一步测试这些模型的适应性，我们特别加入了篇章级自然语言处理任务，来验证它们在长文本上的建模效果。

以下是我们选用的基准测试数据集。

<div align=center>
    <img src="zh-cn/img/bert-wwm/p2.png" /> 
</div>

我们列举其中部分实验结果，完整结果请查看我们的技术报告。为了确保结果的稳定性，每组实验均独立运行10次，汇报性能最大值和平均值（括号内显示）。

关于模型的介绍：

<div align=center>
    <img src="zh-cn/img/bert-wwm/p4.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/bert-wwm/p5.png" /> 
</div>


**中文简体阅读理解：CMRC 2018**

CMRC 2018是哈工大讯飞联合实验室发布的中文机器阅读理解数据。根据给定问题，系统需要从篇章中抽取出片段作为答案，形式与SQuAD相同。

<div align=center>
    <img src="zh-cn/img/bert-wwm/p3.png" /> 
</div>

**中文繁体阅读理解：DRCD**

DRCD数据集由中国台湾台达研究院发布，其形式与SQuAD相同，是基于繁体中文的抽取式阅读理解数据集。

<div align=center>
    <img src="zh-cn/img/bert-wwm/p6.png" /> 
</div>

**中文命名实体识别：人民日报，MSRA-NER**

中文命名实体识别（NER）任务中，我们采用了经典的人民日报数据以及微软亚洲研究院发布的NER数据。

<div align=center>
    <img src="zh-cn/img/bert-wwm/p7.png" /> 
</div>

**句对分类：LCQMC，BQ Corpus**

LCQMC以及BQ Corpus是由哈尔滨工业大学（深圳）发布的句对分类数据集。

<div align=center>
    <img src="zh-cn/img/bert-wwm/p8.png" /> 
</div>

**篇章级文本分类：THUCNews**

由清华大学自然语言处理实验室发布的新闻数据集，需要将新闻分成10个类别中的一个。

<div align=center>
    <img src="zh-cn/img/bert-wwm/p9.png" /> 
</div>

**自然语言推断 XNLI**

<div align=center>
    <img src="zh-cn/img/bert-wwm/p10.png" /> 
</div>

**中文情感分析 ChnSentiCorp**

<div align=center>
    <img src="zh-cn/img/bert-wwm/p11.png" /> 
</div>


### 5.使用建议

基于以上实验结果，我们给出以下使用建议（部分），完整内容请查看paper。

+ 初始学习率是非常重要的一个参数（不论是BERT还是其他模型），需要根据目标任务进行调整。
+ ERNIE的最佳学习率和BERT/BERT-wwm相差较大，所以使用ERNIE时请务必调整学习率（基于以上实验结果，ERNIE需要的初始学习率较高）。
+ 由于BERT/BERT-wwm使用了维基百科数据进行训练，故它们对正式文本建模较好；而ERNIE使用了额外的百度百科、贴吧、知道等网络数据，它对非正式文本（例如微博等）建模有优势。
+ 在长文本建模任务上，例如阅读理解、文档分类，BERT和BERT-wwm的效果较好。
+ 如果目标任务的数据和预训练模型的领域相差较大，请在自己的数据集上进一步做预训练。
+ 如果要处理繁体中文数据，请使用BERT或者BERT-wwm。因为我们发现ERNIE的词表中几乎没有繁体中文。

### 6.声明

虽然我们极力的争取得到稳定的实验结果，但实验中难免存在多种不稳定因素（随机种子，计算资源，超参），故以上实验结果仅供学术研究参考。由于ERNIE的原始发布平台是PaddlePaddle（<https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE>），我们无法保证在本报告中的效果能反映其真实性能（虽然我们在若干数据集中复现了效果）。同时，上述使用建议仅供参考，不能作为任何结论性依据。

该项目不是谷歌官方发布的中文Whole Word Masking预训练模型。


### 7.总结

我们发布了基于全词覆盖的中文BERT预训练模型，并在多个自然语言处理数据集上对比了BERT、ERNIE以及BERT-wwm的效果。实验结果表明，在大多数情况下，采用了全词覆盖的预训练模型（ERNIE，BERT-wwm）能够得到更优的效果。由于这些模型在不同任务上的表现不一致，我们也给出了若干使用建议，并且希望能够进一步促进中文信息处理的研究与发展。