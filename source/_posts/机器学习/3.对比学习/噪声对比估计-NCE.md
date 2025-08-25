---
title: 噪声对比估计-NCE
categories:
  - 机器学习
  - 3.对比学习
date: 2025-08-25 10:08:52
tags:
---

NCE目标函数：
$$\begin{aligned}\max_\theta J^c(\theta)&=\max_\theta \mathbb{E}_X[log P_\theta( D | w , θ)]
\\&=\max_\theta \left(\mathbb{E}_{P(w|c)}\left[\log\frac{P_\theta(w|c)}{P_\theta(w|c)+kP(w)}\right]+k\mathbb{E}_{P(w)}\left[\log\frac{kP(w)}{P_\theta(w|c)+kP(w)}\right] \right)\end{aligned}$$

最早提出NCE思想的论文
[Noise-Contrastive Estimation of Unnormalized Statistical Models-2010](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)
[Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics-2012](https://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf)
给出了具体的NCE算法，本文主要参考来源于此
[A fast and simple algorithm for training neural probabilistic language models-2012](https://arxiv.org/pdf/1206.6426)


回顾一下分布的知识：
设真实数据概率分布的概率密度函数为 $P_d(\cdot)$ ，以下简称分布 $P_d(\cdot)$
机器学习的主要目标是 用一个参数为 $\theta$ 的分布 $P_\theta(\cdot)$ 估计  $P_d(\cdot)$，$P_\theta(\cdot)$称为预测概率分布

>如果能知道$P_\theta(\cdot)$的形式，比如是正态分布或指数分布，那么可以直接学习 $\theta$ 的值
但大部分情况下我们并不知道具体形式，所以是对每个给定数据的估计概率值，也就是直接学习概率分布

概率分布要满足积分为1，即 $\int P(x)dx = 1$

一般情况下，预测概率分布需要通过归一化，来保证满足积分为1的条件
$$P_\theta(\cdot)=\frac{\hat{P}_\theta(\cdot)}{Z_\theta}$$
其中分子是非归一化的概率分布，分母 $Z_\theta$ 是配分函数（Partition Function）也称为归一化常数 （Normalized Constant） 或 Marginalized Evidence 

用神经网络来估计为例
logits 层的输出 是非归一化的概率分布
经过softmax层之后才是 归一化的概率分布


## 1. NCE: Noise Contrastive Estimation

[A fast and simple algorithm for training neural probabilistic language models](https://arxiv.org/pdf/1206.6426)
NCE 是一个机器学习的方法，不涉及神经网络
- 学习一个参数来表示 $Z_\theta$
- 学习一个能区分 从真实数据分布和噪声分布采样数据的模型的模型

假设我们的数据是文本，任务是根据给定的上下文context $c$，预测目标target为单词 $w$ ，希望学习到一个参数为$\theta$（用$\theta$参数化）的预测分布来估计/建模真实分布：
$$P(w|c) \approx P_\theta(w|c)$$
让我们假设预测分布 $P_\theta$ 服从某一个指数族分布，任务是学习该分布的参数$\theta$ 值
$$P_\theta(w|c)\:=\:\frac{\exp\{s_\theta(w,c)\}}{\sum_{w\in V}\exp\{s_\theta(w,c)\}}\:=\:\frac{u_\theta(w,c)}{Z_\theta}$$
$V$为词汇表，$S_θ(w,c)$是参数为 $θ$ 的评分函数，它量化了词$w$与上下文$c$的相容性，一般定义为向量点积

### 1.1. ML method
在机器学习（ML）方法中，是通过最大似然估计（Maximum likelihood estimation,MLE）(假设所有样本之间相互独立)来优化参数 $\theta$，目标函数为最大化对数似然$\log P_\theta(w|c)$的期望：

$$\max_\theta L^c(\theta)=\max_\theta\mathbb{E}_{w\sim P(w|c)}\left[\log P_\theta(w|c)\right]$$
这个期望展开为
$$
\mathbb{E}_{w\sim P(w|c)}\left[\log P_\theta(w|c)\right] = \sum_{w \in V}P(w|c)\log P_\theta(w|c)
$$

对应的损失函数为  负对数似然$\log P_\theta(w|c)$的期望
$$\mathcal{L}_{MLE} = -L^c(\theta)=-\mathbb{E}_{w\sim P(w|c)}\left[\log P_\theta(w|c)\right] = -\sum_{w \in V}P(w|c)\log P_\theta(w|c)$$
可以看到，这个其实就是类别数为 $|V|$ 的多分类交叉熵，

梯度为
$$\begin{aligned}\frac{\partial}{\partial\theta}L^c(\theta)&=\frac{\partial}{\partial\theta} \mathbb{E}_{w\sim P(w|c)}\left[\log P_{\theta}(w|c)\right]\\&=\frac{\partial}{\partial\theta}\mathbb{E}_{w\sim P(w|c)}\left[\log\frac{\exp\{s_\theta(w,c)\}}{Z_\theta}\right]\\
&=\frac{\partial}{\partial\theta}\mathbb{E}_{w\sim P(w|c)}s_\theta(w,c) - \frac{\partial}{\partial\theta} logZ_\theta\\
&=\sum_{w\in V}[P(w|c)-P_\theta(w|c)]\frac{\partial}{\partial\theta}s_\theta(w,c)
\end{aligned}$$

实际计算中，给定一个在上下文 $c$ 中观察到的词 $w$，就对$L^c(\theta)$求一次梯度，P(w|c)只对观察到的词 $w$，为1：
$$
\begin{aligned}\frac{\partial}{\partial\theta}L^c(\theta)&=\sum_{w\in V}[P(w|c)-P_\theta(w|c)]\frac{\partial}{\partial\theta}s_\theta(w,c)\\
&=\frac{\partial}{\partial\theta}s_\theta(w,c)-\sum_{w\in V}\frac{\exp s_\theta(w,c)}{\sum_{w\in V}\exp\{s_\theta(w,c)\}})\frac{\partial}{\partial\theta}s_\theta(w,c)
\end{aligned}
$$

优化他有些困难的，在计算梯度时计算词汇表中所有单词的$s_θ ( w , c)$来求 $P_\theta(w|c)$ 中的$Z_{\theta}$


论文里提到了Importance sampling 来解决 $Z_{\theta}$ 计算复杂度高的问题，但是存在一些缺点。

### 1.2. NCE method
噪声对比估计（Noise-Contrastive Estimation，NCE）:一种参数学习方法

不是通过最大似然估计直接求参数，而是通过对比来求参数，任务是学习一个能区分从真实数据分布和噪声分布采样数据的模型，从而学习到 $P_\theta(w|c)$

这个模型其实就是一个二元分类器 $P_\theta(D|w,c)$ ，来估计$P(D|w,c)$ ，标签D=1或0分别表示 $w$ 是来自真实数据分布 $P(w|c)$ （论文中称为 $P^c_d$ ），还是噪声分布 $P(w)$ （论文中称为 $P_n$ ）

> 二元分类器可以通过逻辑回归来进行学习。

在噪声对比估计中，往往在数据分布 $P(w|c)$ 中采样1个正样本w，标签D=1。然后从噪声分布 $P(w)$ 中采样k个负样本w，标签D=0

也就是说，这k+1个样本构成的样本集$X$来自分布 $\frac{1}{k+1}P(w|c) + \frac{k}{k+1}P(w)$

那么标签D=1，即样本来自真实分布 $P(w|c)$的后验概率为
$$P(D=1|w,c)=\frac{P(w|c)}{P(w|c)+kP(w)}$$

由于我们希望用$P_θ(w|c)$拟合$P(w|c)$，所以我们用$P_θ(w|c)$代替方程中的$P(w|c)$，使后验概率成参数$θ$的函数：
$$P_\theta(D=1|w,c)=\frac{P_\theta(w|c)}{P_\theta(w|c)+kP(w)}$$

我们简单地在真实数据和噪声样本的混合下得到的一个样本集$X$上做优化，最大化对数似然$log P_\theta( D | w , θ)$的期望值
$$\begin{aligned}\max_\theta J^c(\theta)&=\max_\theta \mathbb{E}_X[log P_\theta( D | w , θ)]
\\&=\max_\theta \left(\mathbb{E}_{P(w|c)}\left[\log\frac{P_\theta(w|c)}{P_\theta(w|c)+kP(w)}\right]+k\mathbb{E}_{P(w)}\left[\log\frac{kP(w)}{P_\theta(w|c)+kP(w)}\right] \right)\end{aligned}$$

对$J^c(\theta)$ 求梯度
$$\begin{aligned}\frac{\partial}{\partial\theta}J^c(\theta)&= \frac{\partial} {\partial\theta}\left(\mathbb{E}_{P(w|c)}\left[\log\frac{P_\theta(w|c)}{P_\theta(w|c)+kP(w)}\right]+k\mathbb{E}_{P(w)}\left[\log\frac{kP(w)}{P_\theta(w|c)+kP(w)}\right] \right)
\\ 
&=\mathbb{E}_{P(w|c)}\left[\frac{kP(w)}{P_\theta(w|c)+kP(w)}\frac{\partial} {\partial\theta}\log P_\theta(w|c)\right]-k\mathbb{E}_{P(w)}\left[\frac{P_\theta(w|c)}{P_\theta(w|c)+kP(w)}\frac{\partial} {\partial\theta}\log P_\theta(w|c)\right]\\
&=\sum_{w\in V}(P(w|c)-P_\theta(w|c))\frac{kP(w)}{P_\theta(w|c)+kP(w)}\frac{\partial}{\partial\theta}\log P_\theta(w|c)\end{aligned}$$

当 $k → ∞$，趋近于最大似然的梯度
$$\frac{\partial}{\partial\theta}J^c(\theta)\to\sum_{w\in v}(P(w|c)-P_\theta(w|c))\frac{\partial}{\partial\theta}\log P_\theta(w|c)$$

实际训练过程中，给定一个在上下文$c$中观察到的词$w$，我们通过生成$k$个噪声样本$x_1,\dots,x_k$，$w$对梯度的贡献为
$$\begin{aligned}\frac{\partial}{\partial\theta}J^c(\theta)=&\frac{kP(w)}{P_{\theta}(w|c)+kP_{n}(w)}\frac{\partial}{\partial\theta}\operatorname{log}P_{\theta}(w|c)-\\&\begin{aligned}\sum_{i=1}^k\left[\frac{P_\theta(x_i|c)}{P_\theta(x_i|c)+kP(x_i)}\frac{\partial}{\partial\theta}\log P_\theta(x_i|c)\right]\end{aligned}\end{aligned}$$


注意 $\frac{P_\theta(x_i|c)}{P_\theta(x_i|c)+kP(x_i)}$ 的值一定在0到1之间，不像importance sampling的方法一样会变得方差很大，基于NCE的学习是很稳定的

上文所述的 $J^c(\theta)$ 用于学习对某一个上下文$c$的分布$p(w|c)$，称为局部NCE目标函数

通过使用经验上下文概率P(c)作为权重来组合每个上下文c的NCE目标，定义全局NCE目标函数
$$J(\theta)=\sum_cP(c)J(\theta)$$

### 1.3. Dealing with normalizing constants
如上文所述， $P_\theta(w|c)$ 中的$Z_{\theta}$难以计算。NCE通过避免显式归一化和将$Z_{\theta}$作为要学习的参数处理这一问题。因此，模型被参数化为一个参数为 $\theta^0$ 非归一化分布$P_{θ^0}(w|c)$和一个参数$\phi$用于表示$Z_{\theta}$的对数
$$P_\theta(w|c)=P_{\theta^0}(w|c)\exp(\phi)$$
那么 参数 $\theta = \{\theta^0,\phi\}$
每对一个上下文 $c$ 都需要学习一个对应的 $\phi$，这使得难以扩展到具有大规模上下文的情况。

#### 1.3.1. negative sampling 负采样
论文发现将$Z_{\theta}$固定为1效果也很好，使用 $Z_{\theta}=1$ 时的$J^c(\theta)$ 作为目标函数的方法为称为 负采样。

比如用负采样改进的了word2vec
$$\begin{aligned}P(D=0\mid w,c)&=\frac{1}{u_\theta(w,c)+1}\\P(D=1\mid w,c)&=\frac{u_\theta(w,c)}{u_\theta(w,c)+1}.\end{aligned}$$
$u_\theta(w,c) = \exp\{s_\theta(w,c)\}$

[Notes on Noise Contrastive Estimation and Negative Sampling](https://arxiv.org/pdf/1410.8251)
[[NLP复习笔记] Word2Vec: 基于负采样的 Skip-gram 及其 SGD 训练 - 博客园](https://www.cnblogs.com/MarisaMagic/p/17949927)

### 1.4. 复杂度
假设$c$是上下文大小，$d$是单词特征向量维度，$V$是模型的词汇量大小。

利用公式计算预测表示，NCE和ML学习都需要进行$cd^2$操作。
对于ML，从预测的表示中计算下一个单词的分布大约需要$Vd$个操作。
对于NCE，在k个噪声样本下的分类为正样本的概率大约需要$kd$次操作
由于 k<<|V|，所以NCE大大提升了计算速度

### 1.5. 总结
总结一下，NCE做了两件事
- 更改了目标函数，任务从多分类问题到二分类问题
- 验证了 $Z_{\theta}$ 在基于NCE的训练中可以直接设为1




## 2. 扩展阅读，另一个博主的推导

感觉还是原论文里写的更精炼

[NCE噪声对比估计_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1yqcEeWEyc?spm_id_from=333.788.videopod.sections&vd_source=5b329c82286a01997454e14991ec6231)中对NCE的推导：
在给定 $c$ 的情况下，正负样本的概率分别为
$$\begin{aligned}P(d=1,w|c)\:=P(w|d=1,c)P(d=1|c)&=P(w|d=1,c)P(d=1)\\&=P(w|d=1,c)\frac{1}{1+k}\end{aligned}$$
$$\begin{aligned}P(d=0,w|c)\:=P(w|d=0,c)P(d=0|c)&=P(w|d=0,c)P(d=0)\\&=P(w|d=0,c)\frac{k}{1+k}\end{aligned}$$
通过对 $d$ 求和，可以得到概率 $P(w|c)$
$$\begin{aligned}P(w|c)=\sum_dP(d,w|c)&=P(d=1,w|c)+P(d=0,w|c)\\&=P(w|d=1,c)\frac{1}{1+k}+P(w|d=0,c)\frac{k}{1+k}\end{aligned}$$


噪声对比估计的目标函数不再是最大化对数似然，而是
$$\max\left\{\mathbb{E}_{w\sim P(w|d=1,c)}\left[\log P_\theta(d=1|w,c)\right]+k\mathbb{E}_{w\sim P(d=0|w,c)}\left[\log P_\theta(d=0|w,c)\right]\right\}$$
P(w|d=1,c)}其实就是 正样本的分布P_d  P(w|d=0,c)}噪声分布 P(w|d=0,c)} P_n

展开：
$$\begin{aligned}&\mathbb{E}_{w\sim P(w|d=1,c)}\left[\log P_\theta(d=1|w,c)\right]+k\mathbb{E}_{w\sim P(w|d=0,c)}\left[\log P_\theta(d=0|w,c)\right]\\
&=\mathbb{E}_{w\sim P(w|d=1,c)}\left[\log\frac{P_\theta(w|d=1,c)}{P_\theta(w|d=1,c)+kP(w|d=0,c)}\right]+k\mathbb{E}_{w\sim P(w|d=0,c)}\left[\log\frac{kP(w|d=0,c)}{P_\theta(w|d=1,c)+kP(w|d=0,c)}\right]\\
&= \sum_wP(w|d=1,c)\frac{kP(w|d=0,c)}{P_\theta(w|d=1,c)+kP(w|d=0,c)} \frac{\partial}{\partial\theta}\log P_\theta(w|d=1,c)-\sum_wP(w|d=0,c)\frac{kP_\theta(w|d=1,c)}{P_\theta(w|d=1,c)+kP(w|d=0,c)}\frac{\partial}{\partial\theta}\log P_\theta(w|d=1,c)
\end{aligned}$$
可以证明：
当 $k\rightarrow \infty$ 时，并把 $logZ_\theta(c)$ 当做常数 ，有
$$\begin{aligned}&\frac{\partial}{\partial\theta}\left[\mathbb{E}_{w\sim P(w|d=1,c)}\left[\log P_\theta(d=1|w,c)\right]+k\mathbb{E}_{w\sim P(w|d=0,c)}\left[\log P_\theta(d=0|w,c)\right]\right]\\&=\sum_w\left[P(w|d=1,c)-P_\theta(w|d=1,c)\right]\frac{\partial}{\partial\theta}s_\theta(w,c)\end{aligned}$$
可以发现：
在这个情况下，最大化噪声对比估计 等价与最大化似然

我们可以用蒙特卡洛采样法去近似期望，即从数据分布中采样m个点，然后从噪声分布中采样n个点
$$=\frac{1}{m}\sum_{w}\log\frac{P_{\theta}(w|d=1,c)}{P_{\theta}(w,|d=1,c)+kP(w|d=0,c)}+\frac{k}{n}\sum_{w^{-}}\log\frac{kP(w^-|d=0,c)}{P_{\theta}(w^-|d=1,c)+kP(w^-|d=0,c)}$$

当m=1,n=k。那么为
$$\log\frac{P_\theta(w|d=1,c)}{P_\theta(w|d=1,c)+kP(w|d=0,c)}+\sum_{w_-}\log\frac{kP(w|d=0,c)}{P_\theta(w|d=1,c)+kP(w|d=0,c)}$$


负采样是NCE的一种特殊情况，即让归一化项$Z_{\theta}$固定为常数1且令$kP(w|d=0,c)=1\to P(w|d=0,c)=\frac1k$,

那么
$$\begin{aligned}&\mathbb{E}_{w\sim P(w|d=1,c)}\left[\log P_\theta(d=1|w,c)\right]+k\mathbb{E}_{w\sim P(w|d=0,c)}\left[\log P_\theta(d=0|w,c)\right]\\
&=\frac{1}{m}\sum_{w}\log\frac{\exp\{s_\theta(w,c)\}}{\exp\{s_\theta(w,c)\}+1}+\frac{k}{n}\sum_{w_-}\log\frac{1}{\exp\{s_\theta(w,c)\}+1}\\
&=\frac{1}{m}\sum_{w}\log\frac{\exp\{s_{\theta}(w,c)\}/\exp\{s_{\theta}(w,c)\}}{(\exp\{s_{\theta}(w,c)\}+1)/\exp\{s_{\theta}(w,c)\}}+\frac{k}{n}\sum_{w}\log\frac{1}{\exp\{s_{\theta}(w,c)\}+1}\\
&=\frac{1}{m}\sum_{w}\log\frac{1}{1+\exp\{-s_{\theta}(w,c)\}}+\frac{k}{n}\sum_{w}\log\frac{1}{\exp\{s_{\theta}(w,c)\}+1}\\
&=\frac{1}{m}\sum_{w}\log\sigma(s_{\theta}(w,c))+\frac{k}{n}\sum_{w}\log\sigma(-s_{\theta}(w,c))
\end{aligned}$$

当m=1,n=k,则
$$=\log\sigma(s_\theta(w,c))+\sum\log\sigma(-s_\theta(w,c))$$

loss
$$
-\log\sigma(s_\theta(w,c))-\sum\log\sigma(-s_\theta(w,c))
$$


