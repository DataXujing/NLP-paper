## LDA和lda2vec

<!-- https://zhuanlan.zhihu.com/p/31470216 -->
<!-- https://www.jianshu.com/p/5be5eeb34d35 -->
<!-- https://blog.csdn.net/zhaozhn5/article/details/79150004 -->
<!-- https://www.cnblogs.com/gasongjian/p/7631978.html -->

<!-- https://zhuanlan.zhihu.com/p/57418059 -->

<!-- 共轭分布的解释 https://blog.csdn.net/jteng/article/details/61932891 -->
<!-- https://blog.csdn.net/qq_43549752/article/details/89493327 -->

<!-- https://zhuanlan.zhihu.com/p/57418059 -->

<!-- https://www.sohu.com/a/234584362_129720 -->

在机器学习领域，LDA是两个常用模型的简称：Linear Discriminant Analysis 和 Latent Dirichlet Allocation。本文的LDA仅指代Latent Dirichlet Allocation. LDA 在主题模型中占有非常重要的地位，类似于SVD，pLSA(我们将在后续章节详细介绍）等建模，可用于浅层语义分析，在文本语义分析中是一个很好用的模型。本章主要参考腾讯大佬Rickjin 2013年在网上发布的《LDA数学八卦》和李航老师的《统计学习方法》第二版,帮助大家理解LDA的数学原理，并最后通过一个示例代码讲解如何通过Python调用LDA算法。LDA模型中涉及的数学知识较多，包括Gamma函数，Dirichlet分布，Dirichlet-Multinomial共轭,Gibbs Sampling, Variational Inference,贝叶斯文本建模，pLSA建模，以及LDA文本建模。

### 1.Gamma函数

我们曾学习过如下一个长相奇特的Gamma函数:

$$\Gamma(x)=\int_{0}^{\infty} t^{x-1}e^{-t} dx$$
通过分部积分的方式，可以推导出这个函数有如下的递归形式：
$$\Gamma(x+1)=x\Gamma(x)$$
于是很容易证明，$\Gamma(x)$函数可以当成是阶乘在实数集上的延拓，具有如下性质
$$\Gamma(n)=(n-1)!$$
学习了Gamma函数之后，大家可以考虑这两个问题（答案可以在《LDA数学八卦》中找到)

1. 这个长这么怪异的函数，数学家是如何找到的；
2. 为何定义$\Gamma$函数的时候，不使得这个函数定义满足$\Gamma(n)=n!$.

Gamma函数从诞生开始就被许多数学家进行研究，包括高斯，勒让德，威尔斯特斯拉，柳维尔等等。这个函数在现代数学分析中被深入研究，在概率论中也是无处不在，很多统计分布和这个函数相关。Gamma函数有很多妙用，他不但使得$(1/2)!$的计算有意义，还扩展了很多其他的数学概念，比如Gamma函数可以把函数的导数扩展到分数阶。

Gamma函数在概率统计中频繁现身，众多的统计分布，包括常见的统计学三大分布（t分布，$\chi^2$分布，$F$分布），Beta分布，Dirichlet分布的密度函数公式中都有Gamma函数的身影，当然发生最直接联系的是由Gamma函数变换得到的Gamma分布，
$$Gamma(x|\alpha,\beta)=\frac{\beta^{\alpha}t^{\alpha-1}e^{-\beta t}}{\Gamma(\alpha)}$$
其中$\alpha$称为shape parameter,主要决定了分布曲线的形状，而$\beta$称为rate parameter或者inverse scale parameter,主要决定了曲线有多陡。

<div align=center>
    <img src="zh-cn/img/lda/p1.png" /> 
</div>

Gamma分布在概率统计领域也是夜歌万人迷，众多统计分布和她有密切的关系，指数分布和$chi^2$分布都是特殊的Gamma分布。另外Gamma分布做为先验分布是很强大的，在贝叶斯统计分析中被广泛的用作其他分布的先验。如果把统计分布中的共轭关系类比为人类生活中的情侣关系的话，那么指数分布，Possion分布，正态分布，对数正态分布都可以是Gamma分布的情人，

!> 什么是共轭分布

共轭分布(conjugate distribution)的概率中一共涉及到三个分布：先验、似然和后验，如果由先验分布和似然分布所确定的后验分布与该先验分布属于同一种类型的分布，则该先验分布为似然分布的共轭分布，也称为共轭先验。
比较绕嘴，下面从公式来理一下思路。假设变量$x$服从分布$P(x|\theta)$，其观测样本为$X={x_1,x_2,...,x_m}$，参数$\theta$服从先验分布$\Pi(\theta)$。那么后验分布为
$$P(\theta|X)=\frac{\Pi(\theta)P(X|\theta)}{P(X)}$$
如果后验分布$P(\theta|X)$与先验分布$\Pi(\theta)$是同种类型的分布，则称先验分布$\Pi(\theta)$为似然分布$P(x|\theta)$的共轭分布。

比较常用的几个例子有：高斯分布是高斯分布的共轭分布，Beta分布是二项分布的共轭分布，Dirichlet分布是多项分布的共轭分布。下面对二项分布给出证明。
 共轭分布不仅使求后验分布计算简单，更重要的是保留了先验分布的类型，使概率估计更加准确。


### 2.Beta/Dirichlet分布

统计学是猜测上帝的游戏，当然我们不总是有机会猜测上帝，运气不好的时候旧的揣度魔鬼的心思。有一天你被魔鬼撒旦抓走了，撒旦说：“你们人类很聪明，而我又是很仁慈的，和你玩一个游戏，赢了就可以走，否则灵魂出卖给我。游戏的规则很简单，我有一个磨盒，上面有一个按钮，你每按一下按钮，就均匀的输出一个$[0,1]$之间的随机数，我现在按10下，我手上有10个数，你猜第七大的数是什么，偏离不超过0.01就算对。”你应该怎么猜呢？

从数学的角度抽象一下，上面的游戏描述为

<div align=center>
    <img src="zh-cn/img/lda/p2.png" /> 
</div>

在概率统计学中，均匀分布应该算是潘多拉的魔盒，几乎所有重要的概率分布都可以从均匀分布Uniform(0,1)中生成出来
<div align=center>
    <img src="zh-cn/img/lda/p3.png" /> 
</div>

我们可以推导出Beta分布的定义：

$$f(x)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}$$

这就是一般意义上的Beta分布，可以证明，$\alpha,\beta$取非负实数的时候，这个概率密度函数也都是良定义的。

<div align=center>
    <img src="zh-cn/img/lda/p6.png" /> 
</div>

+ Beta-Binomial共轭

<div align=center>
    <img src="zh-cn/img/lda/p4.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/lda/p5.png" /> 
</div>

这个式子描述的就是Beta-Binomial共轭

<div align=center>
    <img src="zh-cn/img/lda/p7.png" /> 
</div>

可以得到$(X_{(k_1)},X_{(k_1+k_2)})$的联合分布是
$$f(x_1,x_2,x_3)=\frac{\Gamma(n+1)}{\Gamma(k_1)\Gamma(k_2)\Gamma(n-k_1-k_2+1)}x_1^{k_1-1}x_2^{k_2-1}x_3^{n-k_1-k_2}$$
熟悉Dirichlet的同学一眼就看出了上面的分布其实就是3维形式的Dirichlet分布$Dir(x_1,x_2,x_3|k_1,k_2,n-K_1-k_2+1)$,令$\alpha_1=k_1,\alpha_2=k_2,\alpha_3=n-k_1-k_2+1$，于是分布密度可以写为
$$f(x_1,x_2,x_3)=\frac{\Gamma(\alpha_1+\alpha_2+\alpha_3)}{\Gamma(\alpha_1)\Gamma(\alpha_2)\Gamma{\alpha_3}}x_1^{\alpha_1-1}x_2^{\alpha_2-1}x_3^{\alpha_3-1}$$
这就是一般形式的Dirichlet分布，即便$\alpha_i,i=1,2,3$延拓到非负实数集合，以上概率分布也是良定义的。

从形式上我们可以看出Dirichlet分布是Beta分布在高维度上的推广，他和Beta分布一样是个百变星君，密度函数可以展现出多种形态。
<div align=center>
    <img src="zh-cn/img/lda/p8.png" /> 
</div>

魔鬼的游戏继续：

<div align=center>
    <img src="zh-cn/img/lda/p9.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/lda/p10.png" /> 
</div>

以上游戏描述了Dirichlet-Multinomial共轭。

> Beta/Dirichlet分布的期望的性质

<div align=center>
    <img src="zh-cn/img/lda/p11.png" /> 
</div>


### 3.MCMC和Gibbs Samping

#### 1.随机模拟

随机模拟（或统计模拟）方法有一个很酷的别名是蒙特卡罗方法（Monte Carlo Simulation)(笔者研究生阶段发表的一篇论文:[A fiducial p-value approach for comparing heteroscedastic regression models](https://www.tandfonline.com/doi/abs/10.1080/03610918.2016.1255966))就使用了蒙特卡罗方法进行算法的研究。这个方法的发展起源于
20世纪40年代，和原子弹制造的曼哈顿计划密切相关，当时的几个大牛包括乌拉姆，冯.诺依曼,费米，费曼，Nicholas Metropolis,在美国洛斯阿拉莫斯国家实验室研究裂变物质的种子连锁反应的时候，开始使用统计模拟的方法，并在最早的计算机上进行编程实现。

现代的统计模拟方法最早由数学家乌拉姆提出，被Metropolis命名为蒙特卡罗方法，蒙特卡罗是著名额赌场，赌博总是和统计密切关联的，所以这个命名风趣而且贴切，很快被大家接受。说起蒙特卡罗的源头，可以追溯到18世纪，布丰当年用于计算$\pi$的著名的投针试验就是蒙特卡罗模拟实验。统计采样的方法其实统计学家很早就知道了，但在计算机出现以前，随机数生成的成本很高，所以该方法也没有实用价值。随着计算机技术在20世纪后半叶的迅猛发展，随机模拟技术很快进入实用阶段。对那些使用确定算法不可行或不可能解决的问题，蒙特卡罗方法常常为人民带来希望。

<div align=center>
    <img src="zh-cn/img/lda/p12.png" /> 
</div>

统计模拟有个红药的问题就是给定一个概率分布$p(x)$，我们如何在计算机中生成他的样本，一般而言均匀分布U(0,1)的样本是相对容易生成的。通过线性同于发生器可以生成伪随机数，我们用确定性算法生成[0,1]之间的伪随机数序列后，这些序列的各种统计指标和均与分布的理论计算结果非常接近。这样伪随机数序列就有比较好的统计性质，可以被当成真实的随机数使用。

<div align=center>
    <img src="zh-cn/img/lda/p13.png" /> 
</div>

而我们常见的概率分布，无论是连续的还是离散的分布，都可以基于均与分布的样本生成。例如正态分布可以通过著名的Box-Muller变换得到

**定理（Box-Muller变换）** 如果随机变量$U_1,U_2$独立同分布于$Uniform[0,1]$,则
$$Z_0=\sqrt{-2\ln U_1}\cos (2\pi U_2)$$
$$Z_1=\sqrt{-2\ln U_1}\sin (2\pi U_2)$$
则，$Z_0,Z_1$独立同分布于标准正态分布。

其他几个著名的连续分布包括指数分布，Gamma分布，t分布，F分布，Beta分布，Dirichlet分布等等，也都可以通过类似的数学变换得到；离散的分布通过均匀分布更加容易生成，大家可以参考统计计算的书，其中Sheldon M.Ross的《统计模拟》是写的比较通俗易懂的一本。

不过我们并不是总是这么幸运的，当$p(x)$的形式很复杂，或者$p(x)$是个高维的分布的时候，样本的生成就可能很困难了。 譬如有如下的情况

1. $p(x)=\frac{\tilde{p}(x)}{\int \tilde{p}(x)dx}$,而$\tilde{p}(x)$ 我们是可以计算的，但是底下的积分式无法显式计算。
2. $p(x,y)$是一个二维的分布函数，这个函数本身计算很困难，但是条件分布$p(x|y)$的计算相对简单;如果$p(x)$是高维的，这种情形就更加明显。

此时就需要使用一些更加复杂的随机模拟的方法来生成样本。而本节中将要重点介绍的 MCMC(Markov Chain Monte Carlo) 和 Gibbs Sampling算法就是最常用的一种，这两个方法在现代贝叶斯分析中被广泛使用。要了解这两个算法，我们首先要对马氏链的平稳分布的性质有基本的认识。

#### 2.马氏链及其平稳分布

马氏链的数学定义很简单
$$P(X_{t+1}=x|X_t,X_{t-1},...)=P(X_{t+1}=x|X_t)$$
也就是状态转移的概率只依赖于前一个状态。

我们先来看马氏链的一个具体的例子。社会学家经常把人按其经济状况分成3类：下层(lower-class)、中层(middle-class)、上层(upper-class)，我们用1,2,3 分别代表这三个阶层。社会学家们发现决定一个人的收入阶层的最重要的因素就是其父母的收入阶层。如果一个人的收入属于下层类别，那么他的孩子属于下层收入的概率是 0.65, 属于中层收入的概率是 0.28, 属于上层收入的概率是 0.07。事实上，从父代到子代，收入阶层的变化的转移概率如下

<div align=center>
    <img src="zh-cn/img/lda/p14.png" /> 
</div>

使用矩阵的表示方式，转移概率矩阵记为

<div align=center>
    <img src="zh-cn/img/lda/p15.png" /> 
</div>

假设当前这一代人处在下层、中层、上层的人的比例是概率分布向量$\pi_0=[\pi_0(1),\pi_0(2),\pi_0(3)]$，那么他们的子女的分布比例将是$\pi_1=\pi_0P$, 他们的孙子代的分布比例将是$\pi_2=\pi_1P=\pi_0P^2,...$, 第$n$代子孙的收入分布比例将是$\pi_n=\pi_{n-1}P=\pi_0P^n$。

假设初始概率分布为$\pi_0=[0.21,0.68,0.11]$，则我们可以计算前$n$代人的分布状况如下

<div align=center>
    <img src="zh-cn/img/lda/p16.png" /> 
</div>

我们发现从第7代人开始，这个分布就稳定不变了，这个是偶然的吗？我们换一个初始概率分布$\pi_0=[0.75,0.15,0.1]$试试看，继续计算前$n$代人的分布状况如下

<div align=center>
    <img src="zh-cn/img/lda/p17.png" /> 
</div>

我们发现，到第9代人的时候, 分布又收敛了。最为奇特的是，两次给定不同的初始概率分布，最终都收敛到概率分布$\pi=[0.286,0.489,0.225]$，也就是说收敛的行为和初始概率分布$\pi_0$无关。这说明这个收敛行为主要是由概率转移矩阵$P$决定的。我们计算一下$P^n$

<div align=center>
    <img src="zh-cn/img/lda/p18.png" /> 
</div>

我们发现，当$n$足够大的时候，这个$P^n$矩阵的每一行都是稳定地收敛到$\pi=[0.286,0.489,0.225]$这个概率分布。自然的，这个收敛现象并非是我们这个马氏链独有的，而是绝大多数马氏链的共同行为，关于马氏链的收敛我们有如下漂亮的定理：

**定理（马氏链定理）** 如果一个非周期马氏链具有转移概率矩阵$P$,且它的任何两个状态是连通的，那么$\lim_{n \to \infty} P^n_{ij}$存在且与$i$无关，记$\lim_{n\to \infty}P^n_{ij}=\pi(j)$, 我们有

<div align=center>
    <img src="zh-cn/img/lda/p19.png" /> 
</div>

这个马氏链的收敛定理非常重要，所有的 MCMC(Markov Chain Monte Carlo) 方法都是以这个定理作为理论基础的。 定理的证明相对复杂，一般的随机过程课本中也不给证明，所以我们就不用纠结它的证明了，直接用这个定理的结论就好了。我们对这个定理的内容做一些解释说明：

1. 该定理中马氏链的状态不要求有限，可以是有无穷多个的；
2. 定理中的“非周期“这个概念我们不打算解释了，因为我们遇到的绝大多数马氏链都是非周期的；
3. 两个状态$i,j$是连通并非指$i$可以直接一步转移到$j(P_{ij}>0)$,而是指$i$ 可以通过有限的$n$步转移到达$j(P^n_{ij}>0)$。马氏链的任何两个状态是连通的含义是指存在一个$n$, 使得矩阵$P^n$中的任何一个元素的数值都大于零。
4. 我们用$X_i$表示在马氏链上跳转第$i$步后所处的状态，如果$\lim_{n\to \infty}P^n_{ij}=\pi(j)$存在，很容易证明以上定理的第二个结论。


#### 3.Markov Chain Monte Carlo

对于给定的概率分布$p(x)$我们希望能有便捷的方式生成它对应的样本。由于马氏链能收敛到平稳分布， 于是一个很的漂亮想法是：如果我们能构造一个转移矩阵为$P$的马氏链，使得该马氏链的平稳分布恰好是$p(x)$, 那么我们从任何一个初始状态$x_0$出发沿着马氏链转移, 得到一个转移序列 $x_0,x_1,x_2,...,x_n,x_{n+1},...$， 如果马氏链在第$n$步已经收敛了，于是我们就得到了$\pi(x)$的样本$x_n,x_{n+1},...$。

这个绝妙的想法在1953年被 Metropolis想到了，为了研究粒子系统的平稳性质， Metropolis 考虑了物理学中常见的波尔兹曼分布的采样问题，首次提出了基于马氏链的蒙特卡罗方法，即Metropolis算法，并在最早的计算机上编程实现。Metropolis 算法是首个普适的采样方法，并启发了一系列 MCMC方法，所以人们把它视为随机模拟技术腾飞的起点。 Metropolis的这篇论文被收录在《统计学中的重大突破》中， Metropolis算法也被遴选为二十世纪的十个最重要的算法之一。

我们接下来介绍的MCMC 算法是 Metropolis 算法的一个改进变种，即常用的 Metropolis-Hastings 算法。由上一节的例子和定理我们看到了，马氏链的收敛性质主要由转移矩阵$P$决定, 所以基于马氏链做采样的关键问题是如何构造转移矩阵$P$,使得平稳分布恰好是我们要的分布$p(x)$。如何能做到这一点呢？我们主要使用如下的定理。

**定理(细致平稳条件)** 如果非周期马氏链的转移矩阵$P$和分布$\pi(x)$满足
$$\pi(i)P_{ij}=\pi(j)P_{ji}$$
则$\pi(x)$是马氏链的平稳分布，上式被称为细致平稳条件(detailed balance condition)

假设我们已经有一个转移矩阵为$Q$马氏链($q(i,j)$表示从状态$i$转移到状态$j$的概率，也可以写为$q(j|i)$或者$q(i\to j)$), 显然，通常情况下
$$p(i)q(i,j) \neq p(j)q(j,i)$$
也就是细致平稳条件不成立，所以$p(x)$不太可能是这个马氏链的平稳分布。我们可否对马氏链做一个改造，使得细致平稳条件成立呢？譬如，我们引入一个$\alpha(i,j)$, 我们希望
$$p(i)q(i,j)\alpha(i,j) = p(j)q(j,i)\alpha(j,i)$$
取什么样的$\alpha(i,j)$以上等式能成立呢？最简单的，按照对称性，我们可以取
$$\alpha(i,j)=p(j)q(j,i),\alpha(j,i)=p(i)q(i,j)$$
于是上式就成立了。

于是我们把原来具有转移矩阵$Q$的一个很普通的马氏链，改造为了具有转移矩阵Q^{'}的马氏链，而$Q^{'}$恰好满足细致平稳条件，由此马氏链$Q^{'}$的平稳分布就是$p(x)$！这里$Q^{'}$的元素为：
$$Q^{'}(i,j)=q(i,j)\alpha(i,j)$$

在改造$Q$的过程中引入的$\alpha(i,j)$称为接受率，物理意义可以理解为在原来的马氏链上，从状态$i$以$q(i,j)$的概率转跳转到状态$j$的时候，我们以$\alpha(i,j)$的概率接受这个转移，于是得到新的马氏链$Q^{'}$的转移概率为$q(i,j)\alpha(i,j)$。

<div align=center>
    <img src="zh-cn/img/lda/p20.png" /> 
</div>


假设我们已经有一个转移矩阵$Q$(对应元素为$q(i,j)$), 把以上的过程整理一下，我们就得到了如下的用于采样概率分布$p(x)$的算法。

<div align=center>
    <img src="zh-cn/img/lda/p21.png" /> 
</div>


上述过程中$p(x),q(x|y)$说的都是离散的情形，事实上即便这两个分布是连续的，以上算法仍然是有效，于是就得到更一般的连续概率分布$p(x)$的采样算法，而$q(x|y)$就是任意一个连续二元概率分布对应的条件分布。

以上的 MCMC 采样算法已经能很漂亮的工作了，不过它有一个小的问题：马氏链$Q$在转移的过程中的接受率$\alpha$可能偏小，这样采样过程中马氏链容易原地踏步，拒绝大量的跳转，这使得马氏链遍历所有的状态空间要花费太长的时间，收敛到平稳分布$p(x)$的速度太慢。有没有办法提升一些接受率呢?

假设$\alpha(i,j)=0.1,\alpha(j,i)=0.2$, 此时满足细致平稳条件，于是
$$p(i)q(i,j)\times 0.1=p(j)q(j,i)\times 0.2$$
上式两边扩大5倍，我们改写为
$$p(i)q(i,j)\times 0.5=p(j)q(j,i)\times 1$$

看，我们提高了接受率，而细致平稳条件并没有打破！这启发我们可以把细致平稳条件中的$\alpha(i,j)$同比例放大，使得两数中最大的一个放大到1，这样我们就提高了采样中的跳转接受率。所以我们可以取
<div align=center>
    <img src="zh-cn/img/lda/p22.png" /> 
</div>

于是，经过对上述MCMC 采样算法中接受率的微小改造，我们就得到了如下教科书中最常见的 Metropolis-Hastings 算法。

<div align=center>
    <img src="zh-cn/img/lda/p23.png" /> 
</div>

#### 4.Gibbs Sampling

对于高维的情形，由于接受率$\alpha$的存在(通常$\alpha<1$), 以上 Metropolis-Hastings 算法的效率不够高。能否找到一个转移矩阵$Q$使得接受率$\alpha=1$ 呢？我们先看看二维的情形，假设有一个概率分布$p(x,y)$, 考察$x$坐标相同的两个点$A(x_1,y_1),B(x_1,y_2)$，我们发现
$$p(x_1,y_1)p(y_2|x_1)=p(x_1)p(y_1|x_1)p(y_2|x_1)$$
$$p(x_1,y_2)p(y_1|x_1)=p(x_1)p(y_2|x_1)p(y_1|x_1)$$
所以
$$p(x_1,y_1)p(y_2|x_1)=p(x_1,y_2)p(y_1|x_1)$$
即
$$p(A)p(y_2|x_1)=p(B)p(y_1|x_1)$$

基于以上等式，我们发现，在$x=x_1$这条平行于$y$轴的直线上，如果使用条件分布p(y|x_1)做为任何两个点之间的转移概率，那么任何两个点之间的转移满足细致平稳条件。同样的，如果我们在$y=y_1$这条直线上任意取两个点$A(x_1,y_1),C(x_2,y_1)$,也有如下等式
$$p(A)p(x_2|y_1)=p(C)p(x_1|y_1)$$

<div align=center>
    <img src="zh-cn/img/lda/p24.png" /> 
</div>

于是我们可以如下构造平面上任意两点之间的转移概率矩阵$Q$

<div align=center>
    <img src="zh-cn/img/lda/p25.png" /> 
</div>

有了如上的转移矩阵$Q$, 我们很容易验证对平面上任意两点$X,Y$, 满足细致平稳条件
$$p(X)Q(X\to Y)=p(Y)Q(Y\to X)$$
于是这个二维空间上的马氏链将收敛到平稳分布$p(x,y)$。而这个算法就称为Gibbs Sampling算法,是Stuart Geman和Donald Geman这两兄弟于1984年提出来的，之所以叫做Gibbs Sampling 是因为他们研究了Gibbs random field, 这个算法在现代贝叶斯分析中占据重要位置。

<div align=center>
    <img src="zh-cn/img/lda/p26.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/lda/p27.png" /> 
</div>

以上采样过程中，如图所示，马氏链的转移只是轮换的沿着坐标轴$x$轴和$y$轴做转移，于是得到样本$(x_0,y_0),(x_1,y_1),(x_1,y_2),(x_2,y_2),...$马氏链收敛后，最终得到的样本就是$p(x,y)$的样本，而收敛之前的阶段称为 burn-in period。额外说明一下，我们看到教科书上的 Gibbs Sampling 算法大都是坐标轴轮换采样的，但是这其实是不强制要求的。最一般的情形可以是，在$t$时刻，可以在$x$轴和$y$轴之间随机的选一个坐标轴，然后按条件概率做转移，马氏链也是一样收敛的。轮换两个坐标轴只是一种方便的形式。

以上过程很容易推广到高维的情形：

<div align=center>
    <img src="zh-cn/img/lda/p28.png" /> 
</div>


### 4.文本建模


### 5.LDA文本建模


### 6.Example