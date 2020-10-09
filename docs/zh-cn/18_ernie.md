

<div align=center>
    <img src="zh-cn/img/ernie/1/p1.jpg" /> 
</div>

## ERNIE: Enhanced Representation through Knowledge Integration

<!-- https://blog.csdn.net/qq_27590277/article/details/106264041 -->
<!-- https://mbd.baidu.com/newspage/data/landingshare?context=%7B%22nid%22%3A%22news_9847078146682403080%22%7D&isBdboxFrom=1&pageType=1&rs=3899455150&ruk=dOycTSIJne3oxxhcIEX-NQ -->

<!-- https://www.bilibili.com/video/BV1zV411m7JZ?from=search&seid=14547341028839402771 -->

<!-- [1]ERNIE: Enhanced Representation through Knowledge Integration: https://arxiv.org/pdf/1904.09223
[2]Pre-Training with Whole Word Masking for Chinese BERT: https://arxiv.org/pdf/1906.08101
[3]facebook的SpanBERT: https://arxiv.org/pdf/1907.10529
[4]ERNIE-TsingHua: https://arxiv.org/pdf/1905.07129.pdf
[5]ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding: https://arxiv.org/pdf/1907.12412.pdf
[6]SpanBERT: Improving Pre-training by Representing and Predicting Spans: https://arxiv.org/pdf/1907.10529.pdf
[7]MT-DNN: https://arxiv.org/pdf/1901.11504.pdf
[8]baidu offical video: http://abcxueyuan.cloud.baidu.com/#/play_video?id=15076&courseId=15076&mediaId=mda-jjegqih8ij5385z4&videoId=2866&pId=15081&showCoursePurchaseStatus=false&type=免费课程
[9] Life long learning: https://www.youtube.com/watch?v=8uo3kJ509hA
[10] 【NLP】深度剖析知识增强语义表示模型：ERNIE: https://mp.weixin.qq.com/s/Jt-ge-2aqHZSxWYKnfX_zg -->


ERNIE 1.0 是百度在2019年4月的时候，基于BERT模型，做的进一步的优化，在中文的NLP任务上得到了SOTA的结果。

它主要的改进是在**mask的机制**上做了改进，它的mask不是基本的word piece的mask，而是在pre-trainning阶段增加了外部的知识，由三种level的mask组成，分别是basic-level masking（word piece）+ phrase level masking（WWM style） + entity level masking。在这个基础上，借助百度在中文的社区的强大能力，中文的ERNIE还是用了各种异质(Heterogeneous)的数据集。此外为了适应多轮的贴吧数据，所有ERNIE引入了DLM (Dialogue Language Model) task。

百度的论文看着写得不错，也很简单，而且改进的思路是后来各种改进模型的基础。例如说Masking方式的改进，让BERT出现了WWM的版本，对应的中文版本（Pre-Training with Whole Word Masking for Chinese BERT[2]），以及 facebook的SpanBERT[3]等都是主要基于masking方式的改进。

但是不足的是，因为baidu ERNIE1.0只是针对中文的优化，导致比较少收到国外学者的关注，另外百度使用的是自家的paddle paddle机器学习框架，与业界主流tensorflow或者pytorch不同，导致受关注点比较少。


### 1.Knowlege Masking

「Inituition」: 模型在预测未知词的时候，没有考虑到外部知识。但是如果我们在mask的时候，加入了外部的知识，模型可以获得更可靠的语言表示。

> 例如：哈利波特是J.K.罗琳写的小说。单独预测 哈[MASK]波特 或者 J.K.[MASK]琳 对于模型都很简单，但是模型不能学到哈利波特和J.K. 罗琳的关系。如果把哈利波特直接MASK掉的话，那模型可以根据作者，就预测到小说这个实体，实现了知识的学习。

需要注意的是这些知识的学习是在训练中隐性地学习，而不是直接将外部知识的embedding加入到模型结构中（ERNIE-TsingHua[4]的做法），模型在训练中学习到了更长的语义联系，例如说实体类别，实体关系等，这些都使得模型可以学习到更好的语言表达。

首先我们先看看模型的MASK的策略和BERT的区别。

<div align=center>
    <img src="zh-cn/img/ernie/1/p2.png" /> 
</div>

ERNIE的mask的策略是通过三个阶段学习的，在第一个阶段，采用的是BERT的模式，用的是basic-level masking，然后在加入词组的mask(phrase-level masking), 然后在加入实体级别entity-level的mask。如下图

<div align=center>
    <img src="zh-cn/img/ernie/1/p3.png" /> 
</div>

+ basic level masking: 在预训练中，第一阶段是先采用基本层级的masking就是随机mask掉中文中的一个字。
+ phrase level masking: 第二阶段是采用词组级别的masking。我们mask掉句子中一部分词组，然后让模型预测这些词组，在这个阶段，词组的信息就被encoding到word embedding中了。
+ entity level masking: 
在第三阶段， 命名实体，例如说 `人名`，`机构名`，`商品名`等，在这个阶段被mask掉，模型在训练完成后，也就学习到了这些实体的信息。

不同mask的效果:

<div align=center>
    <img src="zh-cn/img/ernie/1/p4.png" /> 
</div>

### 2.Heterogeneous Corpus Pre-training

训练集包括了

+ Chinese Wikepedia
+ Baidu Baike
+ Baidu news
+ Baidu Tieba 注意模型进行了繁简体的转化，以及是uncased

### 3.DLM (Dialogue Language Model) task

对话的数据对语义表示很重要，因为对于相同回答的提问一般都是具有类似语义的，ERNIE修改了BERT的输入形式，使之能够使用多轮对话的形式，采用的是三个句子的组合[CLS]S1[SEP]S2[SEP]S3[SEP] 的格式。这种组合可以表示多轮对话，例如QRQ，QRR，QQR。Q：提问，R：回答。为了表示dialog的属性，句子添加了dialog embedding组合，这个和segment embedding很类似。

DLM还增加了任务来判断这个多轮对话是真的还是假的:

<div align=center>
    <img src="zh-cn/img/ernie/1/p5.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ernie/1/p6.png" /> 
</div>

### 4.NSP+MLM

在贴吧中多轮对话数据外都采用的是普通的NSP+MLM预训练任务。NSP任务还是有的，但是论文中没写，但是git repo中写了用了。

最终模型效果对比bert:


<div align=center>
    <img src="zh-cn/img/ernie/1/p7.png" /> 
</div>

------

## ERNIE 2.0: A CONTINUAL PRE-TRAINING FRAMEWORK FOR LANGUAGE UNDERSTANDING

百度ERNIE2.0 的出现直接刷榜了GLUE Benchmark.

「Inituition」：就像是我们学习一个新语言的时候，我们需要很多之前的知识，在这些知识的基础上，我们可以获取对其他的任务的学习有迁移学习的效果。我们的语言模型如果增加多个任务的话，是不是可以获得更好的效果？事实上，经发现，ERNIE1.0 + DLM任务以及其他的模型，例如Albert 加了sentence order prediction（SOP）任务之后或者SpanBERT: Improving Pre-training by Representing and Predicting Spans[6]在加上了SBO目标之后 ，模型效果得到了进一步的优化，同时MT-DNN[7]也证明了，在预训练的阶段中加入直接使用多个GLUE下游任务（有监督）进行多任务学习，可以得到SOTA的效果。

于是科学家们就在想那一直加task岂不是更强？百度不满足于堆叠任务，而是提出了一个持续学习的框架，利用这个框架，模型可以持续添加任务但又不降低之前任务的精度，从而能够更好更有效地获得词法lexical，句法syntactic，语义semantic上的表达。

<div align=center>
    <img src="zh-cn/img/ernie/2/p1.png" /> 
</div>

百度的框架提出，主要是在ERNIE1.0的基础上，利用了大量的数据，以及先验知识，然后提出了多个任务，用来做预训练，最后根据特定任务finetune。框架的提出是针对life-long learning的，即终生学习，因为我们的任务叠加，不是一次性进行的（Multi-task learning），而是持续学习(Continual Pre-training)，所以必须避免模型在学了新的任务之后，忘记旧的任务，即在旧的任务上loss变高，相反的，模型的表现应该是因为学习了的之前的知识，所以能够更好更快的学习到现有的任务。为了实现这个目的，百度提出了一个包含pre-training 和fine-tuning的持续学习框架.

<div align=center>
    <img src="zh-cn/img/ernie/2/p2.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ernie/2/p3.jpg" /> 
</div>

### 1.Continual Pre-training

**任务的构建**

百度把语言模型的任务归类为三大类，模型可以持续学习新的任务。

+ 字层级的任务(word-aware pretraining task)
+ 句结构层级的任务(structure-aware pretraining task)
+ 语义层级的任务(semantic-aware pretraining task)

**持续的多任务学习**

对于持续的多任务学习，主要需要攻克两个难点：

+ 如何保证模型不忘记之前的任务？

常规的持续学习框架采用的是一个任务接一个任务的训练，这样子导致的后果就是模型在最新的任务上得到了好的效果但是在之前的任务上获得很惨的效果(knowledge retention)。

+ 模型如何能够有效地训练？

为了解决上一个的问题，有人propose新的方案，我们每次有新的任务进来，我们都从头开始训练一个新的模型不就好了。虽然这种方案可以解决之前任务被忘记的问题，但是这也带来了效率的问题，我们每次都要从头新训练一个模型，这样子导致效率很低。


**百度提出的方案sequential multi-task learning**

聪明的你肯定就会想到，为什么我们要从头开始训练一个模型，我们复用之前学到的模型的参数作为初始化，然后在训练不就行了？是的，但是这样子似乎训练的效率还是不高，因为我们还是要每一轮中都要同时训练多个任务，百度的解决方案是，框架自动在训练的过程中为每个任务安排训练N轮。

+ 初始化 optimized initialization

每次有新任务过来，持续学习的框架使用的之前学习到的模型参数作为初始化，然后将新的任务和旧的任务一起训练。

+ 训练任务安排 task allocating

对于多个任务，框架将自动的为每个任务在模型训练的不同阶段安排N个训练轮次，这样保证了有效率地学习到多任务。如何高效的训练，每个task 都分配有N个训练iteration。

> One left problem is how to make it trained more efﬁciently. We solve this problem by allocating each task N training iterations. Our framework needs to automatically assign these N iterations for each task to different stages of training. In this way, we can guarantee the efﬁciency of our method without forgetting the previously trained knowledge


这样做的好处是：

+ 部分任务的语义信息建模适合递进式： 比如ERNIE1.0 突破完形填空,ERNIE2.0 突破选择题，句子排序题等
+ 不断递进更新，就好像是前面的任务都是打基础，有点boosting的意味
+ 顺序学习容易导致遗忘模式（这个可以复习一下李宏毅的视频），所以只适合学习任务之间比较紧密的任务，就好像你今天学了JAVA，明天学了Spring框架，但是如果后天让你学习有机化学，就前后不能够联系起来，之前的知识就忘得快

### 2.Continual Fine-tuning

在模型预训练完成之后，可以根据特定任务进行fine-tuning，这个和BERT一样。


### 3.ERNIE2.0 Model

为了验证框架的有效性，ERNIE2.0 用了多种任务，训练了新的ERNIE2.0模型，然后成功刷榜NLU任务的benchmark，GLUE（截止2020.01.04）。**百度开源了ERNIE2.0英文版，但是截至目前为止，还没有公开中文版的模型**。

**model structure**

模型的结构和BERT一致，但是在预训练的阶段，除了正常的position embedding，segment embdding，token embedding还增加了「task embedding」。用来区别训练的任务, 对于N个任务，task的id就是从`0～N-1`，每个id都会被映射到不同的embedding上。模型的输入就是：

<div align=center>
    <img src="zh-cn/img/ernie/2/p3.png" /> 
</div>

但是对于fine-tuning阶段，ERNIE使用任意值作为初始化都可以。

**Pre-training Tasks**

ERNIE模型堆叠了大量的预训练目标。就好像我们学习英语的时候，我们的卷子上面，有多种不同的题型。

+ 词法层级的任务(word-aware pretraining task)：获取词法知识。
    - knowledge masking(1.0): ERNIE1.0的任务
    - 大小写预测（Capitalization Prediction Task）: 模型预测一个字是不是大小写，这个对特定的任务例如NER比较有用。（但是对于中文的话，这个任务比较没有用处，可能可以改为预测某个词是不是缩写）
    - 词频关系（Token-Document Relation Prediction Task）: 预测一个词是不是会多次出现在文章中，或者说这个词是不是关键词。

+ 语法层级的任务(structure-aware pretraining task) ：获取句法的知识
    - 句子排序(Sentence Reordering Task): 把一篇文章随机分为`i = 1到m份`，对于每种分法都有 $i!$种组合，所以总共有 $\sum_{i=1}^{m}i!$种组合，让模型去预测这篇文章是第几种，就是一个多分类的问题。这个问题就能够让模型学到句子之间的顺序关系。就有点类似于Albert的SOP任务的升级版。
    - 句子距离预测(Sentence Distance Task): (一个三分类的问题：0: 代表两个句子相邻,1: 代表两个句子在同个文章但不相邻,2: 代表两个句子在不同的文章中

+ 语义层级的任务(semantic-aware pretraining task) ：获取语义关系的知识(0: 代表了提问和标题强相关（出现在搜索的界面且用户点击了）,1: 代表了提问和标题弱相关（出现在搜索的界面但用户没点击）,2: 代表了提问和标题不相关（未出现在搜索的界面)
    - 篇章句间关系任务(Discourse Relation Task): 判断句子的语义关系例如logical relationship( is a, has a, contract etc.)
    - 信息检索关系任务(IR Relevance Task): 一个三分类的问题，预测query和网页标题的关系:0: 代表了提问和标题强相关（出现在搜索的界面且用户点击了）,1: 代表了提问和标题弱相关（出现在搜索的界面但用户没点击）,2: 代表了提问和标题不相关（未出现在搜索的界面）

<div align=center>
    <img src="zh-cn/img/ernie/2/p4.jpg" /> 
</div>

**network output**

+ Token-Level loss：给每个token一个label
+ Sequence-Level loss：例如句子重排任务，判断[CLS]的输出是那一类别

### 4.应用场景

**场景：性能不敏感的场景：直接使用**

!> 度小满的风控召回排序提升25%

<div align=center>
    <img src="zh-cn/img/ernie/2/p6.jpg" /> 
</div>

度小满的风控识别上：训练完的ERNIE上直接进行微调，直接预测有没有风险对应的结果，传统的缺点：需要海量的数据，而这些数据也很难抓取到的，抓取这些特征之后呢还要进行复杂的文本特征提取，比如说挖掘短信中银行的催收信息，对数据要求的量很高，对数据人工的特征的挖掘也很高。这两项呢造成了大量的成本，如今只需ERNIE微调一下，当时直接在召回的排序上得到25%的优化。这种场景的特点是什么？对于用户的实时性的需求不是很强，不需要用户输入一个字段就返回结果。只要一天把所有数据得到，跑完，得到结果就可以了，统一的分析就可以了，适合少数据的分析场景。

**场景：性能敏感场景优化：模型蒸馏，例如搜索问答Query识别和QP匹配**

<div align=center>
    <img src="zh-cn/img/ernie/2/p7.jpg" /> 
</div>

另外的一个场景需要非常高的性能优势的，采用的解决方案就是模型蒸馏，是搜索问答query识别和QP匹配，输入一个问题，得到答案，本质是文本匹配，实际是输入问题，把数据库中大量的候选答案进行匹配计算得分，把得分最高的返回。但是百度每天很多用户，很快的响应速度，数据量大，要求响应速度还快，这时候要求不仅模型特别准，而且还要特别快，怎么解决就是模型蒸馏，

+ phrase 1: 判断问题是否可能有答案（文本分类），过滤完是可能有答案的，再与数据库中进行匹配，因为大部分输入框的不一定是个问题，这样过滤掉一部分，排除掉一部分后，在做匹配就能得到很大的提升；提升还是不够，第一部分其实是文本分类，通过小规模的标注特征数据进行微调，得到一个好的模型，同时日志上是有很多没有标注的数据，用ERNIE对这些数据进行很好的标注，用一个更好的模型去标注数据，用这些标注数据训练相对简单的模型，就实现了蒸馏，ERNIE处理速度慢，但是可以用题海战术的方式训练简单的模型。具体步骤：一个很优秀的老师，学一点东西就能够带学生了，但是学生模型不够聪明，海量的题海战术就可以学很好。

1. fine-tune：使用少量的人工标注的数据用ERNIE训练
2. label propagation：使用ERNIE标注海量的挖掘数据，得到带标注的训练数据
3. train：使用这些数据下去训练一个简单的模型或者采用模型蒸馏的方式，参考Tiny-BERT, ERNIE-tiny 。

+ phrase 2: 有答案与答案库进行各种各样的匹配（文本匹配）同理，下面问题匹配也是，右边也是query和答案，然后经过embedding，加权求和，全连接，最后计算他们之间的预选相似度，可以是余弦相似度。召回提升7%

**场景：百度视频离线推荐**

推荐场景：是可以提前计算好，保存好的，可变的比较少，视频本身就是存好的，变化量不会很大，更新也不会特别频繁，离线把相似度计算好，保存起来就可以，两两计算之间的相似度计算量是非常大的，那么怎么减少计算量呢？使用了一个技术叫离线向量化，离线把视频和视频的相似度算好，然后存入数据库 N个视频两两计算其复杂度为$O(N^2)$

+ 采用了离线向量化
+ 用户看的视频经过一个ERNIE 得到一个向量
+ 候选集通过另外一个ERNIE（共享权重），得到一个向量，计算相似度，计算复杂度为$O(N)$，之后再两两计算余弦相似度。

代码参考：

<https://github.com/PaddlePaddle/ERNIE>

<https://github.com/nghuyong/ERNIE-Pytorch>

