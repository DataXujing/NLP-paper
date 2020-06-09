## LDA

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

我们日常生活中总是产生大量的文本，如果每一个文本存储为一篇文档，那每篇文档从人的观察来说就是有序的词的序列$d=(w_1,w_2,...,w_n)$

<div align=center>
    <img src="zh-cn/img/lda/p29.png" /> 
</div>

统计文本建模的目的就是追问这些观察到语料库中的的词序列是如何生成的。统计学被人们描述为猜测上帝的游戏，人类产生的所有的语料文本我们都可以看成是一个伟大的上帝在天堂中抛掷骰子生成的，我们观察到的只是上帝玩这个游戏的结果——词序列构成的语料，而上帝玩这个游戏的过程对我们是个黑盒子。所以在统计文本建模中，我们希望猜测出上帝是如何玩这个游戏的，具体一点，最核心的两个问题是

+ 上帝都有什么样的骰子；
+ 上帝是如何抛掷这些骰子的；

第一个问题就是表示模型中都有哪些参数，骰子的每一个面的概率都对应于模型中的参数；第二个问题就表示游戏规则是什么，上帝可能有各种不同类型的骰子，上帝可以按照一定的规则抛掷这些骰子从而产生词序列。

<div align=center>
    <img src="zh-cn/img/lda/p30.png" /> 
</div>

#### 1.Unigram Model

假设我们的词典中一共有$V$个词 $v_1,v_2,...,v_V$，那么最简单的 Unigram Model 就是认为上帝是按照如下的游戏规则产生文本的。

<div align=center>
    <img src="zh-cn/img/lda/p31.png" /> 
</div>

上帝的这个唯一的骰子各个面的概率记为$\overrightarrow{p}=(p_1,p_2,...,p_V)$ , 所以每次投掷骰子类似于一个抛钢镚时候的贝努利实验， 记为$w\sim  Mult(w|\overrightarrow{p})$。

<div align=center>
    <img src="zh-cn/img/lda/p32.png" /> 
</div>

对于一篇文档$d=\overrightarrow{w}=(w_1,w_2,...,w_n)$, 该文档被生成的概率就是
$$p(\overrightarrow{w})=p(w_1,w_2,...,w_n)=p(w_1)p(w_2)...p(w_n)$$
而文档和文档之间我们认为是独立的， 所以如果语料中有多篇文档$W=(\overrightarrow{w_1},\overrightarrow{w_2},...,\overrightarrow{w_m})$,则该语料的概率是
$$p(W)=p(\overrightarrow{w_1})p(\overrightarrow{w_2})...p(\overrightarrow{w_n})$$
在 Unigram Model 中， 我们假设了文档之间是独立可交换的，而文档中的词也是独立可交换的，所以一篇文档相当于一个袋子，里面装了一些词，而词的顺序信息就无关紧要了，这样的模型也称为**词袋模型(Bag-of-words)**。

假设语料中总的词频是$N$, 在所有的$N$个词中,如果我们关注每个词 $v_i$的发生次数$n_i$，那么$\overrightarrow{n}=(n_1,n_2,...,n_V)$正好是一个多项分布

<div align=center>
    <img src="zh-cn/img/lda/p33.png" /> 
</div>

此时， 语料的概率是

<div align=center>
    <img src="zh-cn/img/lda/p34.png" /> 
</div>

当然，我们很重要的一个任务就是估计模型中的参数$\overrightarrow{p}$，也就是问上帝拥有的这个骰子的各个面的概率是多大，按照统计学家中频率派的观点，使用最大似然估计最大化$p(W)$，于是参数$p_i$的估计值就是
$$\hat{p_i}=\frac{n_i}{N}$$

对于以上模型，贝叶斯统计学派的统计学家会有不同意见，他们会很挑剔的批评只假设上帝拥有唯一一个固定的骰子是不合理的。在贝叶斯学派看来，一切参数都是随机变量，以上模型中的骰子$\overrightarrow{p}$不是唯一固定的，它也是一个随机变量。所以按照贝叶斯学派的观点，上帝是按照以下的过程在玩游戏的

<div align=center>
    <img src="zh-cn/img/lda/p35.png" /> 
</div>

上帝的这个坛子里面，骰子可以是无穷多个，有些类型的骰子数量多，有些类型的骰子少，所以从概率分布的角度看，坛子里面的骰子$\overrightarrow{p}$服从一个概率分布$p(\overrightarrow{p})$，这个分布称为参数$\overrightarrow{p}$的先验分布。

<div align=center>
    <img src="zh-cn/img/lda/p36.png" /> 
</div>

以上贝叶斯学派的游戏规则的假设之下，语料$W$产生的概率如何计算呢？由于我们并不知道上帝到底用了哪个骰子$\overrightarrow{p}$,所以每个骰子都是可能被使用的，只是使用的概率由先验分布$p(\overrightarrow{p})$来决定。对每一个具体的骰子$\overrightarrow{p}$,由该骰子产生数据的概率是 $p(W|\overrightarrow{p})$, 所以最终数据产生的概率就是对每一个骰子$\overrightarrow{p}$上产生的数据概率进行积分累加求和
$$p(W)=\int p(W|\overrightarrow{p})p(\overrightarrow{p})d\overrightarrow{p}$$

在贝叶斯分析的框架下，此处先验分布$p(\overrightarrow{p})$就可以有很多种选择了，注意到
$$p(\overrightarrow{n})=Mult(\overrightarrow{n}|\overrightarrow{p},N)$$
实际上是在计算一个多项分布的概率，所以对先验分布的一个比较好的选择就是多项分布对应的共轭分布，即 Dirichlet 分布

<div align=center>
    <img src="zh-cn/img/lda/p37.png" /> 
</div>

此处，$\Delta(\overrightarrow{\alpha})$就是归一化因子$Dir(\overrightarrow{\alpha})$,即

<div align=center>
    <img src="zh-cn/img/lda/p38.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/lda/p39.png" /> 
</div>

回顾前一个小节介绍的 Drichlet 分布的一些知识，其中很重要的一点就是：

**Dirichlet 先验 + 多项分布的数据 Rendered by QuickLaTeX.com 后验分布为 Dirichlet 分布**

进一步，我们可以计算出文本语料的产生概率为：

<div align=center>
    <img src="zh-cn/img/lda/p40.png" /> 
</div>


#### 2.Topic Model 和 pLSA

以上 Unigram Model 是一个很简单的模型，模型中的假设看起来过于简单，和人类写文章产生每一个词的过程差距比较大，有没有更好的模型呢？

我们可以看看日常生活中人是如何构思文章的。如果我们要写一篇文章，往往是先确定要写哪几个主题。譬如构思一篇自然语言处理相关的文章，可能 40% 会谈论语言学、30% 谈论概率统计、20% 谈论计算机、还有10%谈论其它的主题：

+ 说到语言学，我们容易想到的词包括：语法、句子、乔姆斯基、句法分析、主语…；
+ 谈论概率统计，我们容易想到以下一些词: 概率、模型、均值、方差、证明、独立、马尔科夫链、…；
+ 谈论计算机，我们容易想到的词是： 内存、硬盘、编程、二进制、对象、算法、复杂度…；

我们之所以能马上想到这些词，是因为这些词在对应的主题下出现的概率很高。我们可以很自然的看到，一篇文章通常是由多个主题构成的、而每一个主题大概可以用与该主题相关的频率最高的一些词来描述

以上这种直观的想法由Hoffman 于 1999 年给出的pLSA(Probabilistic Latent Semantic Analysis) 模型中首先进行了明确的数学化。Hoffman 认为一篇文档(Document) 可以由多个主题(Topic) 混合而成， 而每个Topic 都是词汇上的概率分布，文章中的每个词都是由一个固定的 Topic 生成的。下图是英语中几个Topic 的例子。

<div align=center>
    <img src="zh-cn/img/lda/p41.png" /> 
</div>

所有人类思考和写文章的行为都可以认为是上帝的行为，我们继续回到上帝的假设中，那么在 pLSA 模型中，Hoffman 认为上帝是按照如下的游戏规则来生成文本的。

<div align=center>
    <img src="zh-cn/img/lda/p42.png" /> 
</div>

以上pLSA 模型的文档生成的过程可以图形化的表示为

<div align=center>
    <img src="zh-cn/img/lda/p43.png" /> 
</div>

我们可以发现在以上的游戏规则下，文档和文档之间是独立可交换的，同一个文档内的词也是独立可交换的，还是一个 bag-of-words 模型。游戏中的$K$个topic-word 骰子，我们可以记为 $\overrightarrow{\varphi_1},\overrightarrow{\varphi_2}...\overrightarrow{\varphi_K}$, 对于包含$M$篇文档的语料$C=(d_1,d_2,...,d_M)$中的每篇文档$d_m$，都会有一个特定的doc-topic骰子$\overrightarrow{\theta}_ m$，所有对应的骰子记为 $\overrightarrow{\theta}_ 1,\overrightarrow{\theta}_ 2,...,\overrightarrow{\theta}_ M$。为了方便，我们假设每个词$w$都是一个编号，对应到topic-word 骰子的面。于是在 pLSA 这个模型中，第$m$篇文档$d_m$中的每个词的生成概率为

<div align=center>
    <img src="zh-cn/img/lda/p44.png" /> 
</div>

所以整篇文档的生成概率为

<div align=center>
    <img src="zh-cn/img/lda/p45.png" /> 
</div>

由于文档之间相互独立，我们也容易写出整个语料的生成概率。关于pLSA的详细的算法可以参考上一章中对于LSA和pLSA的介绍。


### 5.LDA文本建模

!> 一些符号的说明

+ $M$篇文档语料$C=(d_1,d_2,...,d_M)$
+ 每篇文档为$d_m$,都会有一个特定的doc-topic骰子$\overrightarrow{\theta_m}$,每个骰子有$K$个面(每个面代表一个topic)(给每篇文档选doc-topic骰子是一个多项分布)
+ 选出的doc-topic骰子$\overrightarrow{\theta_m}$对应$K$个topic,每个topic对应一个topic-word骰子$\overrightarrow{\varphi_k}$,$K$个topic对应$\overrightarrow{\varphi_1},\overrightarrow{\varphi_2},...,\overrightarrow{\varphi_K}$个骰子，每个topic-word $\overrightarrow{\varphi_k}$骰子的面表示的是word,因此，基于选出的topic去生成topic-word也是一个多项分布。

#### 1 游戏规则

对于上述的 pLSA 模型，贝叶斯学派显然是有意见的，doc-topic 骰子$\overrightarrow{\theta_m}$和 topic-word 骰子$\overrightarrow{\varphi_k}$都是模型中的参数，参数都是随机变量，怎么能没有先验分布呢？于是，类似于对 Unigram Model 的贝叶斯改造， 我们也可以如下在两个骰子参数前加上先验分布从而把 pLSA 对应的游戏过程改造为一个贝叶斯的游戏过程。由于$\overrightarrow{\theta_m}$和$\overrightarrow{\varphi_k}$都对应到多项分布，所以先验分布的一个好的选择就是Drichlet 分布，于是我们就得到了 LDA(Latent Dirichlet Allocation)模型。

<div align=center>
    <img src="zh-cn/img/lda/p46.png" /> 
</div>

在 LDA 模型中, 上帝是按照如下的规则玩文档生成的游戏的

<div align=center>
    <img src="zh-cn/img/lda/p47.png" /> 
</div>

假设语料库中有$M$篇文档，所有的的word和对应的 topic 如下表示
$$\overrightarrow{w}=(\overrightarrow{w_1},\overrightarrow{w_2},...,\overrightarrow{w_M})$$
$$\overrightarrow{z}=(\overrightarrow{z_1},\overrightarrow{z_2},...,\overrightarrow{z_M})$$
其中$\overrightarrow{w_m}$表示第$m$篇文档中的词，$\overrightarrow{z_m}$表示这些词对应的topic编号
<div align=center>
    <img src="zh-cn/img/lda/p48.png" /> 
</div>

#### 2 物理过程分解

使用概率图模型表示， LDA 模型的游戏过程如图所示。

<div align=center>
    <img src="zh-cn/img/lda/p49.png" /> 
</div>


这个概率图可以分解为两个主要的物理过程：

+ $\overrightarrow{\alpha}\to \overrightarrow{\theta_m}\to z_{m,n}$, 这个过程表示在生成第$m$篇文档的时候，先从第一个坛子中抽了一个doc-topic 骰子\overrightarrow{\theta_m}$, 然后投掷这个骰子生成了文档中第 $n$个词的topic编号z_{m,n}$；
+ $\overrightarrow{\beta}\to \overrightarrow{\varphi_k}\to w_{m,n}|k=z_{m,n}$, 这个过程表示用如下动作生成语料中第$m$篇文档的第$n$个词：在上帝手头的$K$个topic-word 骰子$\overrightarrow{\varphi_k}$中，挑选编号为$k=z_{m,n}$的那个骰子进行投掷，然后生成 word $w_{m,n}$；

理解LDA最重要的就是理解这两个物理过程。 LDA 模型在基于$K$个 topic 生成语料中的$M$篇文档的过程中， 由于是 bag-of-words 模型，有一些物理过程是相互独立可交换的。由此， LDA 生成模型中， $M$篇文档会对应于$M$个独立的 Dirichlet-Multinomial 共轭结构； $K$个 topic 会对应于$K$个独立的 Dirichlet-Multinomial 共轭结构。所以理解 LDA 所需要的所有数学就是理解 Dirichlet-Multiomail 共轭，其它都就是理解物理过程。现在我们进入细节， 来看看 LDA 模型是如何被分解为 $M+K$个Dirichlet-Multinomial 共轭结构的。

由第一个物理过程，我们知道$\overrightarrow{\alpha}\to \overrightarrow{\theta_m}\to z_{m}$表示生成第 $m$篇文档中的所有词对应的topics，显然 $\overrightarrow{\alpha}\to \overrightarrow{\theta_m}$对应于 Dirichlet 分布， $\overrightarrow{\theta_m}\to z_{m}$对应于 Multinomial 分布， 所以整体是一个 Dirichlet-Multinomial 共轭结构；

<div align=center>
    <img src="zh-cn/img/lda/p50.png" /> 
</div>

前文介绍 Bayesian Unigram Model 的小节中我们对 Dirichlet-Multinomial 共轭结构做了一些计算。借助于该小节中的结论，我们可以得到

<div align=center>
    <img src="zh-cn/img/lda/p51.png" /> 
</div>

其中$\overrightarrow{n_m}=(n^{(1)}_ m,...,n^{(K)}_ m)$,$n^{(k)}_ m$表示第$m$篇文档中第$k$个topic产生的词的个数。进一步，利用 Dirichlet-Multiomial 共轭结构，我们得到参数$\overrightarrow{\theta_m}$的后验分布恰好是
$$Dir(\overrightarrow{\theta_m}|\overrightarrow{n_m}+\overrightarrow{\alpha})$$

由于语料中 $M$篇文档的 topics 生成过程相互独立，所以我们得到 $M$个相互独立的 Dirichlet-Multinomial 共轭结构，从而我们可以得到整个语料中 topics 生成概率

<div align=center>
    <img src="zh-cn/img/lda/p52.png" /> 
</div>

目前为止，我们由$M$篇文档得到了$M$个 Dirichlet-Multinomial 共轭结构，还有额外$K$个 Dirichlet-Multinomial 共轭结构在哪儿呢？在上帝按照之前的规则玩 LDA 游戏的时候，上帝是先完全处理完成一篇文档，再处理下一篇文档。文档中每个词的生成都要抛两次骰子，第一次抛一个doc-topic骰子得到 topic, 第二次抛一个topic-word骰子得到 word，每次生成每篇文档中的一个词的时候这两次抛骰子的动作是紧邻轮换进行的。如果语料中一共有 $N$个词，则上帝一共要抛 $2N$次骰子，轮换的抛doc-topic骰子和 topic-word骰子。但实际上有一些抛骰子的顺序是可以交换的，我们可以等价的调整$2N$次抛骰子的次序：前$N$次只抛doc-topic骰子得到语料中所有词的 topics,然后基于得到的每个词的 topic 编号，后$N$次只抛topic-word骰子生成 $N$个word。于是上帝在玩 LDA 游戏的时候，可以等价的按照如下过程进行：

<div align=center>
    <img src="zh-cn/img/lda/p53.png" /> 
</div>

以上游戏是先生成了语料中所有词的 topic, 然后对每个词在给定 topic 的条件下生成 word。在语料中所有词的 topic 已经生成的条件下，任何两个 word 的生成动作都是可交换的。于是我们把语料中的词进行交换，把具有相同 topic 的词放在一起
$$\overrightarrow{w^{'}}=(\overrightarrow{w_{(1)}},...,\overrightarrow{w_{(K)}})$$
$$\overrightarrow{z^{'}}=(\overrightarrow{z_{(1)}},...,\overrightarrow{z_{(K)}})$$
其中，$\overrightarrow{w_{(k)}}$表示这些词都是由第 $k$个 topic 生成的，$\overrightarrow{z_{(k)}}$对应于这些词的 topic 编号.

对应于概率图中的第二个物理过程$\overrightarrow{\beta}\to \overrightarrow{\varphi_k}\to w_{m,n}|k=z_{m,n}$，在 $k=z_{m,n}$的限制下，语料中任何两个由 topic $k$生成的词都是可交换的，即便他们不再同一个文档中，所以我们此处不再考虑文档的概念，转而考虑由同一个 topic 生成的词。考虑如下过程 $\overrightarrow{\beta}\to \overrightarrow{\varphi_k}\to w_{(k)}$，容易看出， 此时$\overrightarrow{\beta}\to \overrightarrow{\varphi_k}$ 对应于 Dirichlet 分布， $\overrightarrow{\varphi_k}\to w_{(k)}对应于 Multinomial 分布， 所以整体也还是一个 Dirichlet-Multinomial 共轭结构；

<div align=center>
    <img src="zh-cn/img/lda/p54.png" /> 
</div>

同样的，我们可以得到

<div align=center>
    <img src="zh-cn/img/lda/p55.png" /> 
</div>

其中$\overrightarrow{n_k}=(n^{(1)}_ k,...,n^{(V)}_ k$,$n^{(t)}_ k$表示第$k$个topic产生的词中word $t$的个数,。进一步，利用 Dirichlet-Multiomial 共轭结构，我们得到参数 $\overrightarrow{\varphi_k}$的后验分布恰好是
$$Dir(\overrightarrow{\varphi_k}|\overrightarrow{n_k}+\overrightarrow{\beta})$$

而语料中 $K$个 topics 生成words 的过程相互独立，所以我们得到 $K$个相互独立的 Dirichlet-Multinomial 共轭结构，从而我们可以得到整个语料中词生成概率

<div align=center>
    <img src="zh-cn/img/lda/p56.png" /> 
</div>

从而

<div align=center>
    <img src="zh-cn/img/lda/p57.png" /> 
</div>

*式子-1：联合分布*

此处的符号表示稍微不够严谨, 向量$\overrightarrow{n_k}, \overrightarrow{n_m}$, 都用$n$表示， 主要通过下标进行区分， $k$下标为 topic 编号, $m$下标为文档编号。

#### 3 Gibbs Sampling

有了联合分布$p(\overrightarrow{w},\overrightarrow{z})$, 万能的 MCMC 算法就可以发挥作用了！于是我们可以考虑使用 Gibbs Sampling 算法对这个分布进行采样。当然由于$\overrightarrow{w}$是观测到的已知数据，只有$\overrightarrow{z}$是隐含的变量，所以我们真正需要采样的是分布 $p(\overrightarrow{z}|\overrightarrow{w})$ 那篇很有名的LDA 模型科普文章 Parameter estimation for text analysis 中，是基于*式子-1：联合分布*式推导 Gibbs Sampling 公式的。此小节中我们使用不同的方式，主要是基于 Dirichlet-Multinomial 共轭来推导 Gibbs Sampling 公式，这样对于理解采样中的概率物理过程有帮助。

语料库$\overrightarrow{z}$中的第$i$个词我们记为$i=(m,n)$,是一个二维下标，对应于第$m$篇文档的第$n$个词，我们用$-i$ 表示去除下标为$i$的词。那么按照 Gibbs Sampling 算法的要求，我们要求得任一个坐标轴$i$对应的条件分布$p(z_i=k|\overrightarrow{z_{-i}},\overrightarrow{w})$ 。假设已经观测到的词$w_i=t$, 则由贝叶斯法则，我们容易得到

<div align=center>
    <img src="zh-cn/img/lda/p58.png" /> 
</div>

由于$z_i=k,w_i=t$只涉及到第$m$篇文档和第i=$k$个 topic，所以上式的条件概率计算中, 实际上也只会涉及到如下两个Dirichlet-Multinomial 共轭结构

1. $\overrightarrow{\alpha}\to \overrightarrow{\theta_m}\to z_{m}$
2. $\overrightarrow{\beta}\to \overrightarrow{\varphi_k}\to w_{k}$

其他$M+K-2$个Dirichlet-Multinomial共轭结构和$z_i=k,w_i=t$是独立的。

由于在语料去掉第$i$个词对应的$(z_i,w_i)$，并不改变我们之前讨论的$M+K$个 Dirichlet-Multinomial 共轭结构，只是某些地方的计数会减少。所以$\overrightarrow{\theta_m},\overrightarrow{\varphi_k}$的后验分布都是 Dirichlet:

<div align=center>
    <img src="zh-cn/img/lda/p59.png" /> 
</div>

使用上面两个式子，把以上想法综合一下，我们就得到了如下的 Gibbs Sampling 公式的推导

<div align=center>
    <img src="zh-cn/img/lda/p60.png" /> 
</div>

以上推导估计是本节中最复杂的式子，表面上看上去复杂，但是推导过程中的概率物理意义是简单明了的：$z_i=k,w_i=t$的概率只和两个 Dirichlet-Multinomail 共轭结构关联。而最终得到的$\hat{\theta}_ {mk},\hat{\varphi}_ {kt}$ 就是对应的两个 Dirichlet 后验分布在贝叶斯框架下的参数估计。借助于前面介绍的Dirichlet 参数估计的公式 ，我们有

<div align=center>
    <img src="zh-cn/img/lda/p61.png" /> 
</div>

于是，我们最终得到了 LDA 模型的 Gibbs Sampling 公式

<div align=center>
    <img src="zh-cn/img/lda/p62.png" /> 
</div>

这个公式是很漂亮的， 右边其实就是 $p(topic|doc).p(word|topic)$，这个概率其实是 $doc\to topic\to word$ 的路径概率，由于topic 有$K$个，所以 Gibbs Sampling 公式的物理意义其实就是在这$K$条路径中进行采样。


<div align=center>
    <img src="zh-cn/img/lda/p63.png" /> 
</div>

#### 4 Training and Inference

有了 LDA 模型，当然我们的目标有两个

+ 估计模型中的参数$\overrightarrow{\varphi_1},\overrightarrow{\varphi_2},...,\overrightarrow{\varphi_K}$和 $\overrightarrow{\theta_1},\overrightarrow{\theta_2},...,\overrightarrow{\theta_M}$；
+ 对于新来的一篇文档$doc_new$，我们能够计算这篇文档的 topic 分布$\overrightarrow{\theta_new}$。

有了 Gibbs Sampling 公式， 我们就可以基于语料训练 LDA 模型，并应用训练得到的模型对新的文档进行 topic 语义分析。训练的过程就是获取语料中的$(z,w)$的样本，而模型中的所有的参数都可以基于最终采样得到的样本进行估计。训练的流程很简单:

<div align=center>
    <img src="zh-cn/img/lda/p64.png" /> 
</div>

对于 Gibbs Sampling 算法实现的细节，请参考 Gregor Heinrich 的 Parameter estimation for text analysis 中对算法的描述，以及LDA(http://code.google.com/p/plda) 的代码实现，此处不再赘述。

由这个topic-word 频率矩阵我们可以计算每一个$p(word|topic)$概率，从而算出模型参数$\overrightarrow{\varphi_1},\overrightarrow{\varphi_2},...,\overrightarrow{\varphi_K}$, 这就是上帝用的$K$个 topic-word 骰子。当然，语料中的文档对应的骰子参数 $\overrightarrow{\theta_1},\overrightarrow{\theta_2},...,\overrightarrow{\theta_M}$ 在以上训练过程中也是可以计算出来的，只要在 Gibbs Sampling 收敛之后，统计每篇文档中的 topic 的频率分布，我们就可以计算每一个 $p(topic|doc)$概率，于是就可以计算出每一个$\overrightarrow{\theta_m}$。由于参数$\overrightarrow{\theta_m}$ 是和训练语料中的每篇文档相关的，对于我们理解新的文档并无用处，所以工程上最终存储 LDA 模型时候一般没有必要保留。通常，在LDA 模型训练的过程中，我们是取 Gibbs Sampling 收敛之后的 $n$个迭代的结果进行平均来做参数估计，这样模型质量更高。

有了 LDA 的模型，对于新来的文档$doc_new$, 我们如何做该文档的 topic 语义分布的计算呢？基本上 inference 的过程和 training 的过程完全类似。对于新的文档， 我们只要认为 Gibbs Sampling 公式中的,$\hat{\varphi}_ {kt}$  部分是稳定不变的，是由训练语料得到的模型提供的，所以采样过程中我们只要估计该文档的 topic 分布$\overrightarrow{\theta_new}$就好了。

<div align=center>
    <img src="zh-cn/img/lda/p65.png" /> 
</div>


### 6.Example

```python
'''
整体过程就是：
首先拿到文档集合，使用分词工具进行分词，得到词组序列；
第二步为每个词语分配ID，既corpora.Dictionary；
分配好ID后，整理出各个词语的词频，使用“词ID：词频”的形式形成稀疏向量，使用LDA模型进行训练
'''

from gensim import corpora, models
import jieba.posseg as jp, jieba
# 文本集
texts = [
    '美国教练坦言，没输给中国女排，是输给了郎平',
    '美国无缘四强，听听主教练的评价',
    '中国女排晋级世锦赛四强，全面解析主教练郎平的执教艺术',
    '为什么越来越多的人买MPV，而放弃SUV？跑一趟长途就知道了',
    '跑了长途才知道，SUV和轿车之间的差距',
    '家用的轿车买什么好']
jieba.add_word('四强', 9, 'n')
flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
stopwords = ('没', '就', '知道', '是', '才', '听听', '坦言', '全面', '越来越', '评价', '放弃', '人') 
words_ls = []
for text in texts:
    words = [word.word for word in jp.cut(text) if word.flag in flags and word.word not in stopwords]
    words_ls.append(words)

# print(words_ls)
# 这是分词过程，然后每句话/每段话构成一个单词的列表，结果如下所示：
# [['美国', '输给', '中国女排', '输给', '郎平'],
# ['美国', '无缘', '四强', '主教练'],...]


#去重，存到字典
dictionary = corpora.Dictionary(words_ls)
# print(dictionary)
# print(dictionary.token2id)
# {'中国女排': 0, '美国': 1, '输给': 2, '郎平': 3, '主教练': 4, 
# '四强': 5, '无缘': 6, '世锦赛': 7, '执教': 8, '晋级': 9,
#  '艺术': 10, 'MPV': 11, 'SUV': 12, '买': 13, '跑': 14, '长途': 15, 
#  '差距': 16, '轿车': 17, '家用': 18}


corpus = [dictionary.doc2bow(words) for words in words_ls]
# print(corpus)
# 按照词ID：词频构成corpus：
# [[(0, 1), (1, 1), (2, 2), (3, 1)],
# [(1, 1), (4, 1), (5, 1), (6, 1)],...]


lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
# print(lda)
# LdaModel(num_terms=19, num_topics=2, decay=0.5, chunksize=2000)

for topic in lda.print_topics(num_words=4):
    print(topic)

# 前面设置了num_topics = 2 所以这里有两个主题，很明显第一个是汽车相关topic，第二个是体育相关topic。
# (0, '0.089"跑" + 0.088"SUV" + 0.088"长途" + 0.069"轿车"')
# (1, '0.104"美国" + 0.102"输给" + 0.076"中国女排" + 0.072"郎平"')


# 主题推断
# print(lda.inference(corpus))
# 上面语料属于哪个主题：
# (array([[5.13748 , 0.86251986],
# [0.6138436 , 4.386156 ],
# [8.315966 , 0.68403417],
# [5.387934 , 0.612066 ],
# [5.3367395 , 0.6632605 ],
# [0.59680593, 3.403194 ]], dtype=float32), None)


for e, values in enumerate(lda.inference(corpus)[0]):
    print(texts[e])
    for ee, value in enumerate(values):
        print('\t主题%d推断值%.2f' % (ee, value))
# 美国教练坦言，没输给中国女排，是输给了郎平
# 主题0推断值0.62
# 主题1推断值5.38
# 美国无缘四强，听听主教练的评价
# 主题0推断值1.35
# 主题1推断值3.65
# 中国女排晋级世锦赛四强，全面解析主教练郎平的执教艺术
# 主题0推断值0.82
# 主题1推断值8.18
# 为什么越来越多的人买MPV，而放弃SUV？跑一趟长途就知道了
# 主题0推断值1.63
# 主题1推断值4.37
# 跑了长途才知道，SUV和轿车之间的差距
# 主题0推断值0.65
# 主题1推断值5.35
# 家用的轿车买什么好
# 主题0推断值3.38
# 主题1推断值0.62


text5 = '中国女排将在郎平的率领下向世界女排三大赛的三连冠发起冲击'
bow = dictionary.doc2bow([word.word for word in jp.cut(text5) if word.flag in flags and word.word not in stopwords])
ndarray = lda.inference([bow])[0]
print(text5)
for e, value in enumerate(ndarray[0]):
    print('\t主题%d推断值%.2f' % (e, value))

# 中国女排将在郎平的率领下向世界女排三大赛的三连冠发起冲击
# 主题0推断值2.40
# 主题1推断值0.60

word_id = dictionary.doc2idx(['体育'])[0]
for i in lda.get_term_topics(word_id):
    print('【长途】与【主题%d】的关系值：%.2f%%' % (i[0], i[1]*100))

# 【长途】与【主题0】的关系值：1.61%
# 【长途】与【主题1】的关系值：7.41%

````


------



## lda2vec

Christopher Moody在2016年初提出的一种新的主题模型算法。

<!-- https://redtongue.github.io/2018/08/27/lda2vev-Mixing-Dirichlet-Topic-Models-and-Word-Embeddings-to-Make-lda2vec/ -->

<!--https://www.sohu.com/a/234584362_129720  -->
<!-- https://github.com/cemoody/lda2vec -->
<!-- https://blog.csdn.net/redtongue/article/details/87873773 -->

<!-- https://blog.csdn.net/u010161379/article/details/51250109 -->

!> 论文地址: <https://arxiv.org/abs/1605.02019>

#### 0.ABSTRACT

已经证明分布式密集词向量在捕捉语言中的标记级语义和句法规则方面是有效的，而主题模型可以在文档上形成可解释的表示。在这项工作中，我们描述了lda2vec，它是一个与Dirichlet分布的主题向量的潜在文档级别混合学习密集词向量的模型。与连续密集的文档表示形式相反，该表达式通过非负单纯形约束产生稀疏的，可解释的文档混合。我们的方法很容易整合到现有的自动分化框架中，并允许无监督的文档表示,适合科学家使用，同时学习单词向量及它们之间的线性关系。

#### 1.Introduction

主题模型因其能够将文档集合组织为一组较小的突出主题而受到欢迎。 与密集的分布式表示形式相反，这些文档和主题表示通常可以被人类理解接受，并且更容易被解释。 这种解释性提供了额外的选项来突出我们的文档系统中的模式和结构。 例如，使用潜在狄利克雷分配（LDA）主题模型可以揭示文档中的词汇集合（Blei et al。，2003），强调时间趋势（Charlin et al。，2015），并推断配套产品的网络（McAuley et al。 。，2015）。 见Blei等人。 （2010年），概述计算机视觉，遗传标记，调查数据和社交网络数据等领域的主题建模。

<div align=center>
    <img src="zh-cn/img/lda/p66.png" /> 
</div>

*图：lda2vec通过将word2vec的skip gram体系结构与Dirichlet优化的稀疏主题混合体相结合，在单词和文档上构建表示。 文中描述了图中出现的各种组件和转换。*

