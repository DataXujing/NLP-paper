## NEZHA: NEURAL CONTEXTUALIZED REPRESENTATION FOR CHINESE LANGUAGE UNDERSTANDING

<!-- https://blog.csdn.net/ljp1919/article/details/103646770 -->
<!-- https://zhuanlan.zhihu.com/p/100044919 -->
<!-- http://blog.itpub.net/69946223/viewspace-2667239/ -->

<!-- https://blog.csdn.net/weixin_43269174/article/details/106255084 -->

### 1.AdamW, LAMB: 大型预训练模型常用优化器

按照时间上的迭代顺序，近些年神经网络先后出现了 Gradient Descent (GD)、Momentum、Adaptive Gradient (AdaGrad)、Root Mean Square prop (RMSprop)、Adaptive Moment estimation (Adam) 等优秀的优化器。到如今，大部分 NLP 预训练模型已不再使用这些方法，而是使用 Adam Weight Decay Regularization (AdamW) 和去年首度亮相的 Layer-wise Adaptive Moments optimizer for Batching training (LAMB)。为何最为传统的 GD，包括衍生的 stochastic GD、mini-batch GD 优化器已不再使用，下文会有详细的介绍。

**Gradient Descent (GD)**

梯度下降法是最为经典的凸优化优化器，思想也非常明确：通过 loss 反向传导计算参数的梯度，参数往哪个方向跑可以让 loss 下降，就让参数往哪个方向更新：

<div align=center>
    <img src="zh-cn/img/nezha/p1.png" /> 
</div>

需要注意的是,$W_k$中的每一个浮点元素的梯度计算和梯度更新，相互之间是完全独立的，这对于理解梯度更新的机理非常重要。上式中， $\alpha$为学习率，通常是一个固定的超参数，学习率越高，收敛越快。但需要注意控制范围。学习率过大，容易造成梯度跨过参数的局部最优点造成参数震荡；学习率过小，会导致训练过程过于漫长。为避免参数震荡，使用 GD 时，学习率通常设置在一个较低值，且训练的 batch_size 越大，学习率越低。梯度裁剪虽能一定程度上解决梯度震荡的问题，但由于输出的概率分布发生偏移，模型收敛也受到一定负面影响，因此需尽可能避免对梯度裁剪的依赖。

**Adaptive Moment estimation (Adam)**

为解决 GD 中固定学习率带来的不同参数间收敛速度不一致的弊端，AdaGrad 和 RMSprop 诞生出来，为每个参数赋予独立的学习率。计算梯度后，梯度较大的参数获得的学习率较低，反之亦然。此外，为避免每次梯度更新时都独立计算梯度，导致梯度方向持续变化，Momentum 将上一轮梯度值加入到当前梯度的计算中，通过某种权重对两者加权求和，获得当前批次参数更新的更新值。 Adam 结合了这两项考虑，既为每一个浮点参数自适应性地设置学习率，又将过去的梯度历史纳入考量：

<div align=center>
    <img src="zh-cn/img/nezha/p2.png" /> 
</div>

实际使用中，通常 $\beta_1=0.9,\beta_2>0.9$。BERT 源代码中，预训练的$\beta_2$为 0.98，微调的$\beta_2$为 0.999，其目的是为了减少对预训练中得到的原始参数结构的破坏，使收敛更为平缓。此外， $m_0$和$v_0$皆为初始化得来，因此训练时参数种子的设置往往对模型结果的影响较大。从上述公式可以看出，训练前期的学习率和梯度更新是比较激进的，到后期逐渐平稳。

虽然 Adam 优化器的使用会导致内存中多出两倍于原参数体量的占用，但与之换来的训练收益使得学术界并没有放弃这一高效的方法。

**Adam Weight Decay Regularization (AdamW)**

Adam 虽然收敛速度快，但没能解决参数过拟合的问题。学术界讨论了诸多方案，其中包括在损失函数中引入参数的 L2 正则项。这样的方法在其他的优化器中或许有效，但会因为 Adam 中自适应学习率的存在而对使用 Adam 优化器的模型失效。AdamW 的出现便是为了解决这一问题，达到同样使参数接近于 0 的目的。具体的举措，是在最终的参数更新时引入参数自身：

<div align=center>
    <img src="zh-cn/img/nezha/p3.png" /> 
</div>

$\lambda$即为权重衰减因子，常见的设置为 `0.005/0.01`。这一优化策略目前正广泛应用于各大预训练语言模型。

**Layer-wise Adaptive Moments optimizer for Batching training (LAMB)**

LAMB 优化器是 2019 年出现的一匹新秀，原论文标题后半部分叫做 “Training BERT in 76 Minutes”，足以看出其野心之大。 LAMB 出现的目的是加速预训练进程，这个优化器也成为 NLP 社区为泛机器学习领域做出的一大贡献。在使用 Adam 和 AdamW 等优化器时，一大问题在于 batch size 存在一定的隐式上限，一旦突破这个上限，梯度更新极端的取值会导致自适应学习率调整后极为困难的收敛，从而无法享受增加的 batch size 带来的提速增益。LAMB 优化器的作用便在于使模型在进行大批量数据训练时，能够维持梯度更新的精度：

<div align=center>
    <img src="zh-cn/img/nezha/p4.png" /> 
</div>


其中， $\phi$是一个可选择的映射函数，一种是$\phi(z)=z$，另一种则为起到归一化作用的$\phi(z)=\min(\max(z, \gamma_l),\gamma_u) $,$\gamma_l$和 $\gamma_u$为预先设定的超参数，分别代表参数调整的下界和上界。这一简单的调整所带来的实际效果非常显著。使用 AdamW 时，batch size 超过 512 便会导致模型效果大幅下降，但在 LAMB 下，batch size 可以直接提到 32,000 而不会导致精度损失。

由于在下游微调预训练模型时，通常无需过大的数据集，因而 LAMB 仅在预训练环节使用。遗憾的是，LAMB 在 batch size 512 以下时无法起到显著作用，目前只能作为大体量财团的工具。华为的nezha在预训练阶段则采用了LAMB优化器。

以下是 LAMB 优化器的 tensorflow1.x 代码：

```python

class LAMBOptimizer(tf.train.Optimizer):
    '''
    LAMBOptimizer optimizer.
    
    # Important Note
        - This is NOT an official implementation.
        - LAMB optimizer is changed from arXiv v1 ~ v3.
        - We implement v3 version (which is the latest version on June, 2019.).
        - Our implementation is based on `AdamWeightDecayOptimizer` in BERT (provided by Google).
    # References
        - LAMB optimier: https://github.com/ymcui/LAMB_Optimizer_TF
        - Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. https://arxiv.org/abs/1904.00962v3
        - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805
    # Parameters
        - There is nothing special, just the same as `AdamWeightDecayOptimizer`.
    '''
    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="LAMBOptimizer"):
        """Constructs a LAMBOptimizer."""
        super(LAMBOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/lamb_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/lamb_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            ############## BELOW ARE THE SPECIFIC PARTS FOR LAMB ##############

            # Note: Here are two choices for scaling function \phi(z)
            # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
            # identity: \phi(z) = z
            # The authors does not mention what is \gamma_l and \gamma_u
            # UPDATE: after asking authors, they provide me the code below.
            # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
            #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

            r1 = tf.sqrt(tf.reduce_sum(tf.square(param)))
            r2 = tf.sqrt(tf.reduce_sum(tf.square(update)))

            r = tf.where(tf.greater(r1, 0.0),
                         tf.where(tf.greater(r2, 0.0),
                                  r1 / r2,
                                  1.0),
                         1.0)

            eta = self.learning_rate * r

            update_with_lr = eta * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

```


### 2.摘要

预训练模型在捕捉深度语境表征方面的成功是有目共睹的。本文提出一个面向中文NLU任务的模型，名为NEZHA(NEural contextualiZed representation for CHinese lAnguage understanding，面向中文理解的神经语境表征模型，哪吒)。NEZHA相较于BERT有如下改进：

(1)函数式相对位置编码

(2)全词覆盖

(3)混合精度训练

(4)训练过程中使用 LAMB 优化器

前两者是模型改进，后两者是训练优化。
NEZHA在以下数个中文自然处理任务上取得了SOTA的结果：

(1)命名实体识别(人民日报NER数据集)

(2)句子相似匹配(LCQMC,口语化描述的语义相似度匹配)

(3)中文情感分类(ChnSenti)

(4)中文自然语言推断(XNLI)


### 3.介绍

预训练模型大部分是基于英文语料(BooksCorpus和English
Wikipedia)，中文界的预训练模型目前现有如下几个：Google的中文BERT(只有base版)、ERNIE-Baidu、BERT-WWM(科大讯飞)。这些模型都是基于Transformer，且在MLM和NSP这两个无监督任务上进行预训练。这些中文模型之间主要差别在于MLM中使用的词遮蔽策略。Google的中文版BERT独立地遮蔽掉每个汉字或WordPiece token。ERNIE-Baidu在MLM中进一步将实体或者短语视为一个整体，如此一个实体或短语包含多个汉字。BERT-WWM的策略相似，称为全词覆盖(Whole Word Masking，WWM)。WWM强制要求所有属于同一中文词的字都要一起被覆盖掉。ERNIE-Baidu 2.0则又额外引入了其他预训练任务，如Token-Document关系预测和句子重排任务。

本文提出的NEZHA整体上是基于BERT的改进。在NEZHA中使用的是函数式相对位置编码，而在原始的Transformer和BERT中每个词使用的是绝对位置编码。位置编码信息直接加到词嵌入作为Transformer的输入。位置编码一般有2种方式：

(1)函数式编码。通过预定义函数(如正弦函数)直接给出位置编码。

(2)参数式编码。此时的位置编码作为模型参数的一部分，通过学习得到。参数式位置编码涉及两个概念，一个是距离，表示这个词离“我”有多远，另一个是维度，Word Embedding一般有几百维，比如512维，每一维有一个值，通过位置和维度两个参数来确定一个位置编码的值。比如有学者提出一种参数相对位置编码方法，将相对位置信息加入到Transformer的自注意力层中。再往后发展二者的结合体，比如Transform-XL和XLNet使用正弦编码矩阵(非学习得到)和两个偏置项(训练学习到的)表示相对位置编码。

本文的NEZHA使用函数式相对位置编码，通过预定义函数的方式在自注意力层编码相对位置。实验结果表明，该方法是一种有效的位置编码方案，并在实验中取得了一致的效果。此外，NEZHA在训练过程中使用了三种已被证明是有效的预训练BERT技术，即全词覆盖，混合精度训练和LAMB优化。

本文的贡献：

(1)系统性研究了大规模中文语料的预训练模型问题

(2)在多个中文NLU任务上评估模型

(3)评估训练因素的有效性，包括位置编码、掩蔽策略、训练语料库源、训练序列的长度。

(4)发布NEZHA模型和源码

### 4.NEZHA 模型

由于NEZHA也是基于BERT，所以关于Transformer和BERT的介绍请参考本教程的相关对应章节进行学习，在此不赘述。

**函数式相对位置编码（Functional Relative Positional Encoding）**


Transformer为了增加模型的并行效率，采用的是Multi-Head Attention机制。虽然Multi-Head Attention相较于RNN可以增加运算效率，但是它丢失了句子中每个token的位置信息。为了使模型更加稳定有效，Transformer和Bert分别在模型中增加了函数式和参数式绝对位置编码，具体如下图所示。

<div align=center>
    <img src="zh-cn/img/nezha/p6.jpg" /> 
</div>


<div align=center>
    <img src="zh-cn/img/nezha/p7.jpg" /> 
</div>

那么问题来了，既然以及有了绝对位置编码，句子中每个token的位置信息已经在模型中有所体现，为什么还要有相对位置编码呢？

那是因为，在BERT模型预训练时，很多数据的真实数据长度达不到最大长度，因此靠后位置的位置向量训练的次数要比靠前位置的位置向量的次数少，造成靠后的参数位置编码学习的不够。在计算当前位置的向量的时候，应该考虑与它相互依赖的token之间相对位置关系，可以更好地学习到信息之间的交互传递。

相对位置编码是如何加入到模型中的呢？


原始Multi-Head Attention是基于Scaled Dot-Product Attention实现的，而Scaled Dot-Product Attention的实现如下图所示，输入的`Q`、`K`和`V`分别由真实输入的序列$x=(x_1,x_2,...,x_n)$乘上不同权重$W^Q$ 、$W^K$和$W^V$得到，输出为序列$z=(z_1,z_2,...,z_n)$长度与输入序列一致。输出$z_i$的计算公式如下：

$$z_{i} = \sum_{j=1}^{n}{\alpha_{ij}(x_{j}W^{V})}$$(公式1)

其中， $\alpha_{ij}$是由位置$i$和位置$j$的隐藏状态求softmax得到，如下：

$$\alpha_{ij} = \frac{exp    e_{ij}}{\sum_{k}^{}{exp    e_{ik}}}$$(公式2)

其中， $e_{ij}$为输入元素的通过$W^Q$和$W^K$变换缩放点积得到，如下：

$$e_{ij} = \frac{(x_{i}W^{Q})(x_{j}W^{K})^{T}}{\sqrt{d_{z}}}$$(公式3)


<div align=center>
    <img src="zh-cn/img/nezha/p8.jpg" /> 
</div>

*Scaled Dot-Product Attention流程图*

!> 上述过程可以参考我们在Transformer章节中的详细介绍！

在相对位置编码方案中，将输出$z_i$加入两个位置之间相对距离的参数，在上述(公式1)和(公式3)中，分别加入两个token的相对位置信息，修改如下得到：

$$z_{i} = \sum_{j=1}^{n}{\alpha_{ij}(x_{j}W^{V} + a_{ij}^{V})}$$(公式4)

$$e_{ij} = \frac{(x_{i}W^{Q})(x_{j}W^{K}+a_{ij}^{K})^{T}}{\sqrt{d_{z}}}$$(公式5)

其中， $\alpha^{V}_ {ij}$和 $\alpha^{k}_ {ij}$ 是位置$i$和位置$j$的相对位置编码，定义$\alpha_ {ij}$位置编码如下

$$a_{ij}[2k]=sin((j-i)/(10000^{\frac{2k}{d_{z}}}))$$

$$a_{ij}[2k+1]=cos((j-i)/(10000^{\frac{2k}{d_{z}}}))  $$


生成相对位置编码embedding代码如下，详见code：

```python


def _generate_relative_positions_matrix(length, max_relative_position, cache=False):
  if not cache:
    range_vec = tf.range(length)
    range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
    distance_mat = range_mat - tf.transpose(range_mat)
  else:
    distance_mat = tf.expand_dims(tf.range(-length+1, 1, 1), 0)
  distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
  final_mat = distance_mat_clipped + max_relative_position
  return final_mat


def _generate_relative_positions_embeddings(length, depth, max_relative_position, name, cache=False):
  relative_positions_matrix = _generate_relative_positions_matrix(
        length, max_relative_position, cache=cache)
  vocab_size = max_relative_position * 2 + 1
  embeddings_table = np.zeros([vocab_size, depth]) #range(vocab_size * depth)#tf.get_variable(name="embeddings", shape=[vocab_size, depth], initializer=create_initializer())

  position = tf.range(0.0, vocab_size, 1.0)#.unsqueeze(1)
  position = tf.reshape(position, [vocab_size, -1])

  for pos in range(vocab_size):
    for i in range(depth // 2):
      embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
      embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

  embeddings_table_tensor = tf.convert_to_tensor(embeddings_table, tf.float32)
  flat_relative_positions_matrix = tf.reshape(relative_positions_matrix, [-1])
  one_hot_relative_positions_matrix = tf.one_hot(flat_relative_positions_matrix, depth=vocab_size)
  embeddings = tf.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
  my_shape = relative_positions_matrix.shape.as_list()
  my_shape.append(depth)

  embeddings = tf.reshape(embeddings, my_shape)
  return embeddings

```


**全词覆盖(WWM)（Whole Word Masking）**

BERT模型通过词掩码的方式实现了双向Transformer。在BERT模型中，被掩住的词是随机挑选的。通过[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)研究表明，将随机掩码词汇替换成全词掩码，可以有效提高预训练模型效果，即如果有一个汉字被掩蔽，属于同一个汉字的其他汉字都被掩蔽在一起。

在初始的BERT中，每个token或者每个汉字都是随机覆盖的。而 NEZHA 预训练模型，则采用了全词覆盖（WWM）策略，当一个汉字被覆盖时，属于同一个汉字所在词的其他汉字都被一起覆盖。该策略被证明比 BERT 中的随机覆盖训练（即每个符号或汉字都被随机屏蔽）更有效。在 NEZHA 的 WWM 实现中，使用了Jieba进行中文分词。在 WWM 训练数据中，每个样本包含多个覆盖汉字，覆盖汉字的总数约占其长度的 12%，随机替换的占 1.5%。尽管这样预测整个词运算难度有所增加，但最终取得的效果更好。

**混合精度训练（Mixed Precision Training）**

在 NEZHA 模型的预训练中采用了混合精度训练技术。该技术可以使训练速度提高2-3倍，同时也减少模型的空间占用，从而可以利用较大的batch size。

在实现混合精度训练时，是在训练过程中的每一个step，为模型的所有weight维护一个FP32的copy，称为Master Weights；在做前向和后向传播过程中，Master Weights会转换成FP16（半精度浮点数）格式，其中权重、激活函数和梯度都是用FP16进行表示，最后梯度会转换成FP32格式去更新Master Weights。

目的：为了提高训练速度，计算float16比float32快。

**优化器改进（LAMB Optimizer）**

LAMB 优化器（上文中有详细的介绍）是专为深度神经元网络大batch size同时分布式训练而设计。尽管使用大的batch size训练可以有效地加快 DNN 训练速度，但是如果不仔细调整学习率，当batch size处理的大小超过某个阈值时，模型的性能可能会受到很大影响。LAMB 优化器则不需要手动调整学习率，而是采用了一种通用的自适应策略。优化器通过使用非常大的batch size(实验中高达 30k 以上)来加速BERT的训练，而不会导致性能损失，甚至在许多任务中获得最先进的性能。值得注意的是，BERT的训练时间最终从3天显著缩短到 76 分钟

NEZHA base模型每个GPU的batch大小为180，large模型每个GPU batch大小为64。


### 5.实验结果

**预训练**

NEZHA预训练数据集：

(1)Chinese Wikipedia

(2)Baidu Baike

(3)Chinese News


<div align=center>
    <img src="zh-cn/img/nezha/p9.png" /> 
</div>

*Table 1 统计了各个预训练模型使用到的数据集情况*

<div align=center>
    <img src="zh-cn/img/nezha/p10.png" /> 
</div>

*Table 2 展示了现有中文预训练模型中用到的预训练技巧*


**实验结果**

评测数据集：CMRC(中文阅读理解2018)、XNLI(跨语言自然语言推理)、LCQMC(大规模中文问题匹配语料)、PD-NER(人民日报命名实体实体数据集)、ChnSenti(中文情感分类数据集)

<div align=center>
    <img src="zh-cn/img/nezha/p11.png" /> 
</div>

*NEZHA在各个数据集fine-tuning的超参数如 Table 3*

<div align=center>
    <img src="zh-cn/img/nezha/p12.png" /> 
</div>

*各个预训练模型在各个数据集上的评测结果如 Table 4 所示*

<div align=center>
    <img src="zh-cn/img/nezha/p13.png" /> 
</div>

*此外还对NEZHA模型中的位置编码、遮蔽策略、训练序列长度和训练语料进行消融研究。消融研究结果如 Table 5 所示*

从消融研究的实验结果可以看出，上述因素都能够促进下游任务性能的提升。对比函数式相对位置编码、参数式绝对位置编码和参数式相对位置编码，可以看出函数式相对位置编码显著优于其他位置编码方式。在CMRC任务中可以看出，相较于相对位置编码使用绝对位置编码真是弱爆了。

### 6.总结

本文提出一个基于中文语料的大规模预训练模型：NEZHA。其中最为重要的贡献在于使用了函数式相对位置编码，该方法显著优于其他位置编码方式。另外，NEZHA模型也集成了几项先进技术：全词覆盖、混合精度训练和LAMB优化器。NEZHA在数个中文自然语言理解任务中取得SOTA结果。

PS：其实这篇文章在9月初已经放出来了，当时没有开放预训练模型，是在走内部的审批流程，直到12月初终于放出来，这时候似乎没了新鲜感。看了下技术点，除了函数式相对位置编码，其他的算是平平无奇吧。

最后我们看一下在CLUE中的排行榜

<div align=center>
    <img src="zh-cn/img/nezha/p14.png" /> 
</div>