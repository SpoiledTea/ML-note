[TOC]

> 2023-09-02

















# chapter 1 Intro

|            | attribute 1 | attribute 2 | label |
| ---------- | ----------- | ----------- | ----- |
| instance 1 |             |             |       |
| instance 2 |             |             |       |
| instance 3 |             |             |       |



the essence of training is to **find a mapping $\mathcal{f}: \mathcal{X}\rightarrow \mathcal{Y} $**

assumption: the samples are drawn from the sample space in an **i.i.d.** way



supervised learning: with label 

​	regression 

​	classification

unsupervised learning: without label

​	clustering



induction = generalization: learned from the samples.

deduction = specialization



"hypothesis space": is a space consisting of all the mappings  $\mathcal{f}: \mathcal{X}\rightarrow \mathcal{Y} $, or, a functional space.

​	in some cases, the dimension of the "hypothesis space" is finite. Then the process of "learning", is to rule out all the hypothesis which are contradicted with what have occurred in the sample space.

​	

inductive bias = feature selection: in favor of particular attributes rather than other.

决策树学习 连接主义学习（手动调参）



# Chapter 2 Performance measurement 

overfitting: 

​	the learner performs way too well in the training set

​	the learning ability is so strong that mistakenly believes some specific characters of the training set also fit for the whole sample space.

​	weak generalization 

underfitting:

​	far from being well-learned.

underfitting is easy to deal with, overfitting otherwise.



NP: 不知道存不存在一个多项式时间内的算法，但是能在多项式时间内验证答案的正确性

P: 存在一个多项式时间内的算法



To test the performance of a learner, we need to appraise the generalization error

​	thus we need to divide the dataset into two parts, one is training set, the other is testing set.



Method 1: **hold-out**

$$D = S \ \cup T, S \ \cap T = \empty$$

in this case, one needs to sample in proportion,which is **stratified sampling** 

Method 2: **cross validation( k-fold cross validation)**

$$D =  \bigcup\limits_{i=1}^k\ D_{i}, D_{i}\cap D_{j} =\empty,\forall i\neq j $$

![](D:\USTC\00 library\machine learning\1693053044509.png)

specification: **Leave-One-Out**

​	well-trained but costly

Method 3: **Bootstrapping**

​	pick a sample from $D$ and place it in $D'$ , repeat for times to obtain a training set $D’$ having the same size with $D$

​	good in small dataset.



performance measure

​	for regression model: 

​		mean squared error $E=E[Y-f(X)]^2$

​	for classification model: 

​		discrete: $E(f;D)=\frac{1}{m} \sum\limits_{i=1}^m \mathbb{1} \{f(x_{i})\neq y_{i}\}$

​		continuous: $E(f;D)=\int\limits_{\vec{x}}\mathbb{1}\{f(\vec{x}) \neq y\}p(\vec{x})d\vec{x}$



Other evaluation method:

**confusion matrix**:
$$
\begin{bmatrix}
&The\ prediction&The\ prediction\\
The\ fact &Positive& Negative\\
True &True\ positive(取真)& False\ negative(弃真)&recall(查全)\\
False&False\ positive(取伪)& True\ negative(弃伪)\\
&precision(查准)&
\end{bmatrix}
$$


$precision = \frac{True \ positive}{True \ positive + False\ positive}$: 在所有判断为“真”的判断中，有多少是正确的。

$recall = \frac{True \ positive}{True \ positive + False \ negative}$：在所有原本为“真”的案例中，有多少是被判断出来了的。

precision up: be conservative

recall up: be aggressive



**Break-Event point(BEP)**: where precision = recall

**F1**: $\frac{1}{F1}=\frac{1}{2}(\frac{1}{precision}+\frac{1}{recall})$

**macro-P,macro-R,macro-F 1 ; micro-P,micro-R,micro-F 1**



**ROC and AUC:**

[ROC and AUC]:[【面试看这篇就够了】如何理解ROC与AUC - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/349366045)

​	**TPR(recall):** 在真实情况为真的全体中，有多少被判断成真了。（越高越好）
$$
TPR=\frac{True\ Positive}{All\ Positives}
$$
​	**FPR:** 在真实情况为假的全体中，多少被判断成真了。（越小越好）
$$
FPR=\frac{False\ Positive}{All\ Negatives}
$$
ROC and AUC are implemented to evaluate the performance of a trained learner:

​	when the training is **done**, given a certain sample, the learner is gonna output a predicted value. 

​	**given a threshold**, which indicates when to assign the predicted value to Group 1 or Group 2, then the FPR and TPR are set as well.

​	**different threshold determines a different pair of (FPR and TPR), thus forming the ROC curve**



**Cost matrix:**
$$
\begin{bmatrix}
&The\ prediction&The\ prediction\\
The\ fact &Positive& Negative\\
True &True\ positive(取真,no\ cost)& False\ negative(弃真, cost_{1})&recall(查全)\\
False&False\ positive(取伪,cost_{2})& True\ negative(弃伪,no\ cost)\\
&precision(查准)&
\end{bmatrix}
$$

let $D = D^+\ \cup D^-$, where $D^+$indicates all the sample $\vec{x}$ in input space $ \mathcal{X}$  which has a **"True" label**, while $D^-$ the opposite.

then we calculate the sensitive cost as follow:
$$
Error = \frac{1}{|D|}\sum\limits_{\vec{x_{i}}\in D^+}\sum\limits_{\vec{x_{j}}\in D^-}\{\mathbb{1}\{f(\vec{x_{i}}) \neq y_{i})\}cost_{1} + \mathbb{1}\{f(\vec{x_{j})\neq y_{j}}\}cost_{2}\}
$$
**hypothesis test**



# Chapter 3 Regression

## least square method

given a instance $\boldsymbol{x}=(x_1;x_2;\dots;x_d)$

linear model generates a $f$:
$$
f(\boldsymbol{x})=\boldsymbol{w}^T\boldsymbol{x}+b
$$
In the case that $\boldsymbol{x}$ is a **scalar**, then we have

| x     | y     |
| ----- | ----- |
| $x_i$ | $y_i$ |

and
$$
f(x)=wx+b
$$


by minimizing 
$$
E_{(w,b)}=\sum\limits_{i=1}^m(y_i-wx_i-b)^2
$$
we get 
$$
(w^*,b^*)=arg\min\limits_{(w,b)}\sum\limits_{i=1}^m(y_i-wx_i-b)^2
$$
**least square method**

****

In the case that $\boldsymbol{x}$ is a **vector**, then we have 

|                      | $x_{*1}$ | $\dots$ | $x_{*d}$ | y     |
| -------------------- | -------- | ------- | -------- | ----- |
| $\boldsymbol{x}_{i}$ | $x_{i1}$ | $\dots$ | $x_{id}$ | $y_i$ |

and 
$$
f(\boldsymbol{x})=\boldsymbol{w}^T\boldsymbol{x}+b
$$
suppose we have **m** instances and for each instance we have **d** attributes, then we can incorporate the constant **b** into the vector $\boldsymbol{w}$ to get 
$$
\hat{\boldsymbol{w}} = (\boldsymbol{w},b)=(w_1,w_2,\dots,w_d,b)
$$
and so 
$$
f(\boldsymbol{x})=\hat{\boldsymbol{w}}^T(\boldsymbol{x},1)
$$
the least square error is written by:
$$
\begin{align}
Error&=\sum\limits_{i=1}^m (f(\boldsymbol{x}_i)-y_i)^2\\ &=\sum\limits_{i=1}^m(y_i-\hat{\boldsymbol{w}}^T(\boldsymbol{x}_i,1))^2 \\&=[y_1-\hat{\boldsymbol{w}}^T(\boldsymbol{x}_1,1),\dots,y_m-\hat{\boldsymbol{w}}^T(\boldsymbol{x}_m,1)]\begin{bmatrix}
y_1-\hat{\boldsymbol{w}}^T(\boldsymbol{x}_1,1)\\
\dots\\
y_m-\hat{\boldsymbol{w}}^T(\boldsymbol{x}_m,1)
\end{bmatrix}\\
\begin{bmatrix}
y_1-\hat{\boldsymbol{w}}^T(\boldsymbol{x}_1,1)\\
\dots\\
y_m-\hat{\boldsymbol{w}}^T(\boldsymbol{x}_m,1)
\end{bmatrix}&=\begin{bmatrix}
y_1\\y_2\\.\\.\\.\\y_m
\end{bmatrix}-\begin{bmatrix}
x_{11},x_{12},\dots,x_{1d},1\\
x_{21},x_{22},\dots,x_{2d},1\\
\dots\\
x_{m1},x_{m2},\dots,x_{md},1\\
\end{bmatrix}_{m\times(d+1)}
\begin{bmatrix}
w_1\\w_1\\.\\.\\.\\w_d\\b\\
\end{bmatrix}_{(d+1)\times 1}\\
&=\begin{bmatrix}
y_1\\y_2\\.\\.\\.\\y_m
\end{bmatrix}-\begin{bmatrix}
\boldsymbol{x}_1^T,1\\
\boldsymbol{x}_2^T,1\\
.\\.\\.\\\boldsymbol{x}_m^T,1
\end{bmatrix}_{m\times(d+1)}\begin{bmatrix}
w_1\\w_1\\.\\.\\.\\w_d\\b\\
\end{bmatrix}_{(d+1)\times 1}\\
&=\boldsymbol{y}-\boldsymbol{X}\hat{\boldsymbol{w}}\\

Error&=(\boldsymbol{y}-\boldsymbol{X}\hat{\boldsymbol{w}})^T(\boldsymbol{y}-\boldsymbol{X}\hat{\boldsymbol{w}})

\end{align}
$$


to minimize the Error, we have 
$$
\hat{\boldsymbol{w}^*}=arg\min\limits_{\hat{\boldsymbol{w}}}(\boldsymbol{y}-\boldsymbol{X}\hat{\boldsymbol{w}})^T(\boldsymbol{y}-\boldsymbol{X}\hat{\boldsymbol{w}})\\
$$
differentiation to obtain (we need $\boldsymbol{X}^T\boldsymbol{X}$ to be full-rank) :
$$
\hat{\boldsymbol{w}^*}=(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}
$$

**generalized linear model**:
$$
y=g^{-1}(\boldsymbol{w}^T\boldsymbol{x}+b)
$$


****

## logistic regression

​	let:
$$
g^{-1}(x)=\frac{1}{1+e^{-x}}
$$

>  we put **-x** instead of **x** to make $g^{-1}(x)$ to be increasing.

​	so we map the predicted value from $\mathbb{R}$ to $[0,1]$, once the **threshold** between$[0,1]$ is set, we turn a **classification problem** into a **regression problem** 

​	not also can this method group the value, but also gives out the **probability distribution**.

****

And now we are going to figure out the some $\boldsymbol{w}$ and $b$ , such that **the predicted label $\hat{y}$ is closet to the true label $y$ in some sense given $\boldsymbol{x}$** , we are going to use **maximum likelihood method**

> **RECALL: maximum likelihood method**
>
> we know two things:
>
> **a)** several random samples are drawn, in a  $\boldsymbol{i.i.d}$  way, from some population $\boldsymbol{X}$, which comes from a density distribution family $\boldsymbol{f(\boldsymbol{x};\theta)}$ with the parameter  $\theta$ unknown.
>
> **b)** and we have the observed value of these sample.
>
> the probability that these observations occurred is:
> $$
> \begin{align}
> P(X_1=x_1,\dots,X_m=x_m)&=f(x_1,\dots,x_m)\\
> &=\prod\limits_{i=1}^mf(x_i;\theta)\\
> \end{align}
> $$
>  so we assumed that, the parameter $\theta$ should be the value that maximize this probability, which is equivalent to:
> $$
> \boldsymbol{l(\theta)=\sum\limits_{i=1}^mlnf(x_i;\theta)}\\
> \boldsymbol{\hat{\theta}=arg \max\limits_{\theta \in \mathcal{H}}\{l(\theta\}}
> $$

 in this case: 

| samples            | attribute 1 | attribute 2 | ...  | attribute 3 | label |
| ------------------ | ----------- | ----------- | ---- | ----------- | ----- |
| $\boldsymbol{x_1}$ | $x_{11}$    | $x_{12}$    | ...  | $x_{1d}$    | $y_1$ |
| $\boldsymbol{x_2}$ | $x_{21}$    | $x_{22}$    | ...  | $x_{2d}$    | $y_2$ |
| ...                | ...         | ...         | ...  | ...         | ...   |
| $\boldsymbol{x_m}$ | $x_{m 1}$   | $x_{m 2}$   | ...  | $x_{md}$    | $y_m$ |

here we have $y_i\in \{0,1\}$

​	and we believe that these $x_{1},...,x_{m}$ are the observed samples drawn from some population $\boldsymbol{X}$

and $y_1,...y_m$ are drawn from some population $\boldsymbol{Y}$ , and we believed that the two random variables are connected:
$$
\begin{align}
\boldsymbol{Y}&=\frac{1}{1+e^{-\boldsymbol{w^TX}+b}}\\
&=f(X;\boldsymbol{\beta})\\
\boldsymbol{\beta}&=(\boldsymbol{w}^T,b)


\end{align}
$$
by showing:
$$
ln\frac{y}{1-y}=\boldsymbol{w}^T\boldsymbol{x}+b
$$
==we believe that: （这个我没懂是为什么）== 
$$
ln\frac{P(Y=1|\boldsymbol{x})}{P(Y=0|\boldsymbol{x})}=\boldsymbol{w}^T\boldsymbol{x}+b
$$
and we believe that:
$$
Y \sim \left(\begin{array}{cc}
0&1\\
P(\boldsymbol{Y}=0|\boldsymbol{x})&P(\boldsymbol{Y}=1|\boldsymbol{x})\\
\end{array} \right)\\
\begin{align}
note\ that: P(\boldsymbol{Y}=1|\boldsymbol{x})&=\frac{1}{1+e^{\boldsymbol{w}^T\boldsymbol{x}+b}}=f_1(\beta)\\
P(\boldsymbol{Y}=0|\boldsymbol{x})&=\frac{e^{\boldsymbol{w}^T\boldsymbol{x}+b}}{1+e^{\boldsymbol{w}^T\boldsymbol{x}+b}}=f_0(\beta)
\end{align}
$$
so  the probability of observing these $y_1,...y_m$ is: 
$$
\begin{align}
P(Y_1=y_1,...,Y_m=y_m)&=f(y_1,\dots,y_m)\\
&=\prod\limits_{i=1}^mf(y_i;\theta)\\
&=\prod\limits_{i=1}^m y_i^{f_1(\beta)}(1-y_i)^{f_2(\beta)}

\end{align}
$$

> 书上的表示是$\sum\limits_{i=1}^m (y_if_1(\beta)+(1-y_i)f_2(\beta))$, 但本质是一回事

therefore:
$$
\boldsymbol{l(\beta)}=\sum\limits_{i=1}^m \{f_1(\beta)\ln y_i+f_2(\beta)\ln (1-y_i)\}\\
\boldsymbol{\hat{\beta}=arg \max\limits_{\beta \in \mathcal{H}}{l(\beta)}}
$$

****

## LDA: linear discriminant analysis

[]:[线性判别分析LDA原理及推导过程（非常详细） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/79696530)

​	we have a dataset, with samples are categorized into several groups. By implementing LDA, we map the samples into a lower dimension sample space, with **the distance of the samples in the same group to be the smallest**, while **the distance of different group to be the largest**.(组内距离最小（方差），组间距离最大。)

> ​	the essence of **variance** is **the average of distance of a several given samples, in Euclidean sense**:
>
> In scalar case:
> $$
> x_1,x_2,...,x_m\\
> \begin{align}
> var&=\sum\limits_{i=1}^m\frac{(x_i-\bar{x})^2}{m}\\
> mean&=\sum\limits_{i=1}^m\frac{x_i}{m}\\
> \end{align}
> $$
> In vector case:
> $$
> \boldsymbol{x_1,x_2,...,x_m},\forall\ i, \boldsymbol{x_i}=(x_{i1},x_{i2},...x_{id})\\
> \begin{align}
> var&=\sum\limits_{i=1}^m\frac{\boldsymbol{||x_i-\bar{x}||^2}}{m}\\
> mean&=\sum\limits_{i=1}^m\frac{\boldsymbol{x_i}}{m}
> \end{align}
> $$
> 

​	In order to do this, we need to find a **lower dimensional space** on which the samples are mapped, thus **the deduction of the dimension** of the dataset is completed.

> **ATTENTION:**
>
> ​	we know how many categories there are, and which category each sample falls into.



**In "K groups" case**

| samples            | attribute 1 | attribute 2 | ...  | attribute 3 | label |
| ------------------ | ----------- | ----------- | ---- | ----------- | ----- |
| $\boldsymbol{x_1}$ | $x_{11}$    | $x_{12}$    | ...  | $x_{1d}$    | $y_1$ |
| $\boldsymbol{x_2}$ | $x_{21}$    | $x_{22}$    | ...  | $x_{2d}$    | $y_2$ |
| ...                | ...         | ...         | ...  | ...         | ...   |
| $\boldsymbol{x_m}$ | $x_{m 1}$   | $x_{m 2}$   | ...  | $x_{md}$    | $y_m$ |

here we have $y_i\in \{0,1\}$

Denote:
$$
\begin{align}
D&=\{\boldsymbol{x_i},y_i\}^m=\{x_{i1},x_{i2},...,x_{id},y_i\}^m\\
D_j&=\{\boldsymbol{x_i}:y_i=j\}, j\in\{a_1,a_2,...a_k\}\\
N_j&=|D_j|\\
\mu_j&=\frac{1}{N_j}\sum\limits_{\boldsymbol{x}_i\in \boldsymbol{D}_j}\boldsymbol{x}_i\\
\mu&=\frac{1}{|D|}\sum\limits_{\boldsymbol{x_i}\in D}\boldsymbol{x_i}

\end{align}
$$
And we are going to map all the samples **from a d-dimension space into a 1-dimension line**, so we need to figure out the **direction vector $\boldsymbol{w}$ (which is also a unit vector) of the line**

​	given a sample $\boldsymbol{x_i}$ , one denotes its projection as $\widetilde{\boldsymbol{x_i}}=(\boldsymbol{x_i}^T\boldsymbol{w})\boldsymbol{w}$

> The projection of $\vec{a}$ on $\vec{b}$:
> $$
> \frac{|\vec{a}|\cos{\theta}\cdot\vec{b}}{|\vec{b}|}=(\vec{a},\vec{b})\cdot \frac{\vec{b}}{|\vec{b}|^2} = \vec{a}^T\vec{b}\cdot \frac{\vec{b}}{|\vec{b}|^2}
> $$

​	We calculate the **over distance within the jth group**
$$
\begin{align}
\sum\limits_{\boldsymbol{x_i}\in D_j}||\widetilde{\boldsymbol{x_i}}-\widetilde{\boldsymbol{\mu_j}}||^2&=\sum\limits_{\boldsymbol{x_i}\in D_j}\boldsymbol{(\widetilde{x_i}-\widetilde{\mu_j})}^T\boldsymbol{(\widetilde{x_i}-\widetilde{\mu_j})}\\&=\boldsymbol{w}^T(\frac{\sum\limits_{\boldsymbol{x_i}\in D_j}\boldsymbol{x_ix_i^T}}{N_j}-\boldsymbol{\mu_j\mu_j^T})\boldsymbol{w}\\
&=\boldsymbol{w}^T\frac{1}{N_j}\sum\limits_{\boldsymbol{x_i}\in D_j}(\boldsymbol{x_i-\mu_j})(\boldsymbol{x_i-\mu_j})^T\boldsymbol{w}\\
\\

(S_w)_{d \times d}&\stackrel{def}{=}\sum\limits_{j=1}^	K\sum\limits_{\boldsymbol{x_i}\in D_j}(\boldsymbol{x_i-\mu_j})(\boldsymbol{x_i-\mu_j})^T\\


\end{align}
$$
​	then we calculate the **over distance between the groups**
$$
\begin{align}
||\boldsymbol{\widetilde{\mu}_j-\widetilde{\mu}}||^2&=(\boldsymbol{\widetilde{\mu}_j-\widetilde{\mu}})^T(\boldsymbol{\widetilde{\mu}_j-\widetilde{\mu}})\\
&=\boldsymbol{w}^T(\boldsymbol{\mu_j-\mu})(\boldsymbol{\mu_j-\mu})^T\boldsymbol{w}\\
\\
(S_b)_{d\times d}&\stackrel{def}{=}\sum\limits_{j=1}^K(\boldsymbol{\mu_j-\mu})(\boldsymbol{\mu_j-\mu})^T

\end{align}
$$
​		and so all we need to do is to find $\boldsymbol{w}$ to minimize $S_w$ while to maximize $S_b$
$$
\begin{align}
\boldsymbol{w^*}&=arg \{\max{S_b},\min{S_w}\}\\
\boldsymbol{J(w)}&\stackrel{def}{=}\frac{\boldsymbol{w^TS_bw}}{\boldsymbol{w^TS_ww}}\\
\\
\boldsymbol{w^*}&=arg\max{\boldsymbol{J(w)}}
\end{align}
$$
​	 we implement the **Lagrange Multipliers**:
$$
\begin{align}
\boldsymbol{w^TS_ww}&=1\\
L(\boldsymbol{w},\lambda)&=\boldsymbol{w^TS_bw}+\lambda(1-\boldsymbol{w^TS_ww})\\
\frac{\partial{L}}{\partial{\boldsymbol{w}}}&=0\\
\boldsymbol{S_bw}&=\lambda\boldsymbol{S_ww}\\
\boldsymbol{S_w^{-1}S_bw}&=\lambda\boldsymbol{w}\\
\end{align}
$$
​	we calculate the **largest characteristic vector** of $\boldsymbol{S_w^{-1}S_b}$ to get $\boldsymbol{w}$

> **DISCUSSION**:
>
> 1. LDA reduces the dimension of the groups to at most K-1 (supposed K groups in total)
>
> $$
> rank(\boldsymbol{S_w^{-1}S_b}) \leq K-1
> $$
>
> 2. when we get $\boldsymbol{w}$ , given a new sample, what we need to do is to project it on  $\boldsymbol{w}$, and observe the position of the projection to determine whether it is closer to group 1 or group 2. (a threshold is needed)

****

**other method for K groups LDA:**

Method 1: One vs One

​	we use LDA for  $$\left(\begin{array}{cc}K\\2\end{array}\right)$$ times, and we obtain$$\left(\begin{array}{cc}K\\2\end{array}\right)$$ learners. And we implement the same new sample toward $$\left(\begin{array}{cc}K\\2\end{array}\right)$$ learners simultaneously, to determine which group it should fall into by vote.

Method 2: One vs Rest

​	we use LDA for K times, and we obtain K learners. By this method, we choose one group to be **the positive** group while the rest of all to be negative. Once we get the K learners,  we observe which outcome gives out a **positive** answer when we implement the new sample into each of the learners.

****

**class-imbalance:**

​	too much negative label over positive label (for instance: $m^+=10,m^-=80$)

​	we implement **threshold-moving**:
$$
\frac{y}{1-y}>\frac{m^+}{m^-}
$$

****

# Chapter 4 Tree

 the basic algorithm of decision tree 

![](D:\USTC\00 library\Machine learning\tree.png)

> ​	there are three instances in which a recursion occurs:
>
> 1. all the sample belongs to the same group
>
>    e.g. : the lables $y_i$ are the same.
>
> 2. have no attribute or the samples are taking the same value among all the attributes.
>
> 3. have no sample.
>
>    every node is a **slice** of the dataset D

## feature selection

**fine the BEST attribute to be the first divider.**

​	the information entropy in a dataset D is defined as below:
$$
\begin{align}
Ent(D)&\stackrel{def}{=}-\sum\limits_{k=1}^{|\mathcal{Y}|}p_k \log_{2}p_k\\
p_k&=\frac{|D_k|}{|D|}
\end{align}
$$
​	the **smaller** the entropy is, the **purer** the dataset is .

​	we have two methods to choose the attributes on which we are going to base.

**Method 1: Information gain**

​	we suppose attribute **a** has V possible values to take, $\{a^1,a^2,...,a^V\}$ , $D^v$ is a slice of $D$ with the value on attribute **a** be taken as $\boldsymbol{a^v}$. **Information gain** is defined as below:
$$
Gain(D,a)=Ent(D)-\sum\limits_{v=1}^V\frac{|D^v|}{|D|}Ent(D^v)
$$

> ​	$Ent(D)$ 是原本数据集的信息熵，而$Ent(D^v)$是以a这个属性作为节点划分之后的新节点上的信息熵，考虑到$D^v$ 是$D$ 的一个切片，因此需要乘以权重，因此二者做差的意思就是：
>
> ​	**以属性a作为节点来作为划分依据所能带来的信息熵的削减程度。**

$$
a*=arg \max\limits_{a\in A} Gain(D,a)
$$

​	such algorithm is in favor of the attributes which have more values to take. in order to fix this, sometimes we implement **gain ratio**
$$
\begin{align}
Gain\_ratio(D,a)&=\frac{Gain(D,a)}{IV(a)}\\
IV(a)&=-\sum\limits_{v=1}^V\frac{|D|^v}{|D|}\log_2\frac{|D|^v}{|D|}

\end{align}
$$
**Method 2: Gini index**\
$$
\begin{align}
Gini(D)&=\sum\limits_{k=1}^{|\mathcal{Y}|}\sum\limits_{k' \neq k}p_kp_{k'}\\
&=1-\sum\limits_{k=1}^{|\mathcal{Y}|}p_{k}^2
\end{align}
$$

> ​	基尼指数是指：如果我随机从数据集里面选任意两个样本，他们的类别不同的概率。如果基尼指数越小，数据集的纯度就越高，就越趋近于是同一个类别。

​	the gini index of the attribute **a** is defined as below:
$$
Gini\_index(D,a)=\sum\limits_{v=1}^V\frac{|D^v|}{|D|}Gini(D^v)
$$
​	so the attribute we are going to find is given out as follow:
$$
a_*=arg \min\limits_{a \in A} Gini\_index(D,a)
$$

****

## pruning

​	we **prune some branches** to avoid **overfitting**.

**Method 1: prepruning** : 

> 1. 在长出新的分支之前，会计算这颗决策树”有这个分支“和”没有这个分支“的泛化性能，泛化性能的评价方式参照之前的做法，比如可以计算一下“有了这根分支的精度”和“没有这根分支的精度”，来决定要不要长出这根分支。
> 2. 存在欠拟合风险，因为他要不要长出这个新分支只靠这目前这一个节点“长”或者“不长”来决定，有可能目前长这个分支的收益是下降的，但是这个分支往下走了之后，收益是提高的。

**Method 2: postpruning**

> 1. 一棵树全部长完了之后，再对叶结点之上的节点进行考虑，比较“剪除之后的泛化性能“和”不剪除的泛化性能“来决定剪不剪除。自下而上地剪。
> 2. 耗时长

****

## What about the continuous value ?

> ​	suppose we are given a dataset $D$ and one of its continuous attributes $\boldsymbol{a}$, which taking values among $\{a_1,a_2,...a_m\}$
>
> ​	we need to determine how to do discretization to split the series of values into two groups. 

​	every type of separation is determined by the chosen **division point $\boldsymbol{t}$** .
$$
T_a=\{\frac{a^i+a^{i+1}}{2}|1\leq i\leq m-1\}
$$
​	by the performance method of information gain, we choose the division point which gives us the greatest reduction of information entropy:
$$
\begin{align}
t^*&=arg \max\limits_{t \in T_a} Gain(D,a,t)\\
&=arg \max\limits_{t \in T_a} \{Ent(D)-\frac{|D_t^+|}{|D|}Ent(D_t^+)-\frac{|D_t^-|}{|D|}Ent(D_t^-)\}

\end{align}
$$

****

## What about the missing value ?

> 1. how to choose the attribute to divide our dataset.
>
>    **we define a modified information gain**
>
> 2. how to decide where a sample belongs to if the sample has missing value on the attribute ? 
>
>    **the simple will fall into all branches according to its weights**

​	$D$ is a dataset, $\widetilde{D}$ is the subset of $D$ with no missing value on attribute $a$. Suppose $a$ taken values among $\{a^1,a^2,...,a^V \}$, and label $\boldsymbol{y}$ are taken value among $\{1,2,3,...,|\mathcal{Y}|\}$

​	we assign a weight $w_{\boldsymbol{x}}$among each sample $\boldsymbol{x}$
$$
\begin{align}
\widetilde{D}^v&=\{\boldsymbol{x}\in \widetilde{D}| a=a^v\}\\
\widetilde{D}_k&=\{\boldsymbol{x}\in \widetilde{D}| y=k\}\\
\widetilde{D}&=\bigcup\limits_{k=1}^{|\mathcal{Y}|}\widetilde{D}_k=\bigcup\limits_{v=1}^V \widetilde{D}^v\\
\\
\rho&=\frac{\sum\limits_{\boldsymbol{x}\in \widetilde{D}}w_{\boldsymbol{x}}}{\sum\limits_{\boldsymbol{x}\in D}w_{\boldsymbol{x}}}\\
\\

\widetilde{p}_k&=\frac{\sum\limits_{\boldsymbol{x}\in \widetilde{D}_k}w_{\boldsymbol{x}}}{\sum\limits_{\boldsymbol{x}\in \widetilde{D}}w_{\boldsymbol{x}}}\\
\\

\widetilde{r}_v&=\frac{\sum\limits_{\boldsymbol{x}\in \widetilde{D}^v}w_{\boldsymbol{x}}}{\sum\limits_{\boldsymbol{x}\in \widetilde{D}}w_{\boldsymbol{x}}}\\
\end{align}
$$

> ​	$\widetilde{p}_k$ is used to calculate the **information entropy.**
>
> ​	$\widetilde{r}_v$ is used to **determine the weight** on each branch, when a sample have missing value on attribute $a$.

$$
\begin{align}
Gain(D,a)&=\rho\times Gain(\widetilde{D},a)\\
&=\rho\times (Ent(\widetilde{D})-\sum\limits_{v=1}^V\widetilde{r}_vEnt(\widetilde{D}^v))\\
Ent(\widetilde{D})&=-\sum\limits_{k=1}^{|\mathcal{Y}|}\widetilde{p}_k
\log_2{\widetilde{p}_k}\\

\end{align}
$$

​	**if the sample has value on attribute $a$, then falls into the according branch with existing weight $\boldsymbol{w_x}$,  otherwise falls into all branches, with weight: $\boldsymbol{\widetilde{r}_k\cdot w_x}$** 

****

**multivariate decision tree**



## Recall:

​	决策树的过程是在选分支，造分支，对原始训练集进行切片。产生分支的本质是在：减小信息熵，从某种程度上来说，提纯了数据集。信息熵变成了属性选择的标准。

​	决策树可以在既有的属性里面选最佳属性，这个时候样本点所落在的样本空间以这些属性为正交轴，如果在这些正交轴来选属性，那就是直接做平行于轴的切片。

​	高级一点，可以不这样切，斜着切，那就是说选取某些特征的线性组合来形成一个新的特征，他的具体物理意义是模糊的，但有可能根据这个特征来做分类或者回归，泛化性能会更好。

# Chapter 5 Neuron network

**M-P neuron model:**

![](D:\USTC\00 library\Machine learning\M-P neuron.png)

> 1. here sometimes we use: 
>
> $$
> f(x) = \left\{\begin{array}{rl} 1&\text{if} \ x\geq 0\\
> 0&\text{if} \ x<0
> \end{array} \right.
> $$
> or for smoothness, we use:
> $$
> f(x)=\frac{1}{1+e^{-x}}
> $$

​	the learning mechanism is to modify the weight $\boldsymbol{w_i}$ and threshold $\boldsymbol{\theta}$ based on each predicted outcome:
$$
\begin{align}
\hat{y}&=f(\sum\limits_{i=1}^nw_ix_i-\theta)\\
\Delta{w_i}&=\eta(y-\hat{y})x_i\\
\\
w_i&\leftarrow w_i+\Delta{w_i}

\end{align}
$$

> 1. $\eta$ is learning rate.
> 2. $(y-\hat{y})$ is a measurement of error.
> 3. **every time the neuron make a mistake, the weight will be modified.**

****

## BP algorithm

**error BackPropagation (BP algorithm)**

​	used to train **multi-layer feedforward neural networks** like this:

![](D:\USTC\00 library\Machine learning\BP.png)

> 1. we have a dataset $\boldsymbol{D=\{(x_i,y_i)\}_{i=1}^m},\boldsymbol{x_i} \in \mathbb{R}^d,\boldsymbol{y_i}\in \mathbb{R}^l$
> 2. the **threshold** of $\boldsymbol{y_j}$ is $\boldsymbol{\theta_j}$ ,while the threshold of $\boldsymbol{b_j}$ is $\boldsymbol{\gamma_j}$ 
> 3. the **connection weight** between $\boldsymbol{b_h}$ to $\boldsymbol{y_j}$ is $\boldsymbol{w_{hj}}$ ,between $\boldsymbol{x_i}$ to $\boldsymbol{b_h}$ is $\boldsymbol{v_{ih}}$
> 4. this network is determined by $\boldsymbol{d\times q+q\times l +l +q}$ parameters. 

​	the mechanism of **BP algorithm**

> **BEAR IN MIND**:
> $$
> \text{for sample $(\boldsymbol{x_k,y_k})=(x_1^k,...,x_d^k,y_1^k,...,y_l^k)$}\\
> \begin{align}
> \hat{y}_j^k&=f(\beta_j-\theta_j) \text{...第k个示例的第j个预测值}\\
> \beta_j&=\sum\limits_{h=1}^qw_{hj}b_h \text{...隐层对第j个输出层神经元的输入}\\
> b_h&=f(\alpha_h-\gamma_h)\text{...第h个隐层神经元的输出}\\
> \alpha_h&=\sum\limits_{i=1}^dv_{ih}x_i\text{...输入层对第h个隐层神经元的输入}\\
> 
> E_k&=\frac{1}{2}\sum\limits_{j=1}^l(\hat{y}_j^k-y_j^k)^2\text{...平方意义下的性能度量}\\
> \end{align}
> $$
> 

​	we modify the **connection weight** by **the gradient of square error**

> ​	the gradient indicate a direction on which a function increases **the most quickly.**

$$
\begin{align}
\Delta{w_{hj}}&= -\eta \frac{\partial E_k}{\partial w_{hj}}\\
&=-\eta \frac{\partial E_k}{\partial \hat{y}_{j}^k} \frac{\partial \hat{y}_{j}^k}{\partial \beta_{j}} \frac{\partial \beta_{j}}{\partial w_{hj}}\\
&=-\eta (\hat{y}_j^k-y_j)f'(\beta_j-\theta_j)b_h\\
&=-\eta (\hat{y}_j^k-y_j)f(\beta_j-\theta_j)(1-f(\beta_j-\theta_j))b_h\\
&=\eta(y_j-\hat{y}_j^k)\hat{y}_j^k(1-\hat{y}_j^k)b_h\\
&=\eta \ g_j\ b_h\\
g_j&\stackrel{def}{=}(y_j-\hat{y}_j^k)\hat{y}_j^k(1-\hat{y}_j^k)\\
\\

\Delta\theta_j&=-\eta\frac{\partial E_k}{\partial \theta_{j}}\\
&=-\eta\frac{\partial E_k}{\partial \hat{y}_{j}^k}\frac{\partial \hat{y}_{j}^k}{\partial \theta_{j}}\\
&=...\\
&=-\eta\ g_j\\
\\

\Delta{v_{ih}}&=-\eta \frac{\partial E_k}{\partial v_{ih}}\\
&=-\eta \sum\limits_{j=1}^{l}\frac{\partial E_k}{\partial \hat{y}_{j}^k} \frac{\partial \hat{y}_{j}^k}{\partial \beta_{j}}
\frac{\partial \beta_{j}}{\partial b_{h}}
\frac{\partial b_{h}}{\partial \alpha_{h}}
\frac{\partial \alpha_{h}}{\partial v_{ih}}\\
&=...\\
&=\eta \ b_h(1-b_h)\sum\limits_{j=1}^{l}w_{hj}g_j \ x_i\\
&=\eta \ e_h\ x_i\\
e_h&\stackrel{def}{=}b_h(1-b_h)\sum\limits_{j=1}^{l}w_{hj}g_j\\
\\
\Delta\gamma_{h}&=-\eta \frac{\partial E_k}{\partial \gamma_{h}}\\
&=-\eta \sum\limits_{j=1}^{l}\frac{\partial E_k}{\partial \hat{y}_{j}^k}
\frac{\partial \hat{y}_{j}^k}{\partial \beta_{j}}
\frac{\partial \beta_{j}}{\partial b_{h}}
\frac{\partial b_{h}}{\partial \gamma_{h}}\\
&=...\\
&=-\eta \ e_h\\
\end{align}
$$

​	so the modification is done as below:
$$
\begin{align}
w_{hj}&=w_{hj}+\Delta{w_{hj}}\\
\theta_{j}&=\theta_{j}+\Delta{\theta_{j}}\\
v_{ih}&=v_{ih}+\Delta{v_{ih}}\\
\gamma_{h}&=\gamma_{h}+\Delta{\gamma_{h}}
\end{align}
$$

> **DISCUSSION:**
>
> 1. how to set the number of the **hidden layer**.
> 2. sometimes we used $E=\sum\limits_{k=1}^mE_k$ to replace $E_k$ .
> 3. overfitting happens.
> 4. $E(\boldsymbol{w_{hj},v_{ih},\theta_j,\gamma_h})$ is a function with respect to these parameters, and so the learning mechanism can be seen as a process of figuring out **the minimum value** of the function.

****

**other neuron network:**

​	**RBF**

****

**Deep learning:**

​	multiple of hidden layers

​	every layer functions as **feature selection**

​	CNN: 卷积神经网络

## Recall:

​	神经网络的训练是在训练连接权和阈值，这里面的数学工具是激活函数和层数一多起来的链式求导

​	隐层的存在让神经网络变得比较复杂，隐层可以理解成是：我对粗糙的输入层，做了一个特征的提取，从某种程度上也是一种信息熵的减小和信息的纯化。

​	但不同于决策树的学习，他是对训练集中的单个示例，每个示例都跑一遍**“为了最小化均方误差而修改连接权和阈值”**的过程，在这个过程里，两个不同示例的先后顺序会否影响泛化结果？

​	还有问题就是：隐层要多少层才合适，学习率要怎么设置才比较合适？

****

# Chapter 6 Support vector machine

## SVM

​	if there exits a good plane classifying every sample properly. we use SVM to find the best plane. 
$$
\boldsymbol{w}^T\boldsymbol{x}+b=0\\
\text{the distance between the sample to the plane is:}\\
r=\frac{|\boldsymbol{w}^T\boldsymbol{x}+b|}{||\boldsymbol{w}||}\\
\text{for}\ (\boldsymbol{x_i},y_i)\in D:\\
y_i=+1,\boldsymbol{w}^T\boldsymbol{x}+b>0;\\
y_i=-1,\boldsymbol{w}^T\boldsymbol{x}+b<0\\
\\
\text{let:}
\left \{ \begin{array}{cc}
\boldsymbol{w}^T\boldsymbol{x}+b \geq+1,&y_i=+1\\
\boldsymbol{w}^T\boldsymbol{x}+b \leq-1,&y_i=-1\\
\end{array} \right.\\

\text{support vector makes the equality holds}\\
\text{the distance between two support vector is:}\\
\frac{2}{||\boldsymbol{w}||}\\
$$
​	the best plane has the **maximum margin** , the whole point is to minimize $||\boldsymbol{w}||$ which is equivalent to minimize $||\boldsymbol{w}||^2$ under some constraints:
$$
\begin{align}
\min\limits_{\boldsymbol{w},b}&\ \frac{1}{2}||\boldsymbol{w}^2||\\
s.t.& \ \ y_i(\boldsymbol{w}^T\boldsymbol{x}+b)\geq 1,i=1,2,...,m.
\end{align}
$$
​	by lagrange multiplers:
$$
\begin{align}
L(\boldsymbol{w},b,\boldsymbol{\alpha})&=\frac{1}{2}||\boldsymbol{w}^2||+\sum\limits_{i=1}^m\alpha_i(1-y_i(\boldsymbol{w}^T\boldsymbol{x}+b))\\
\frac{\partial L}{\partial \boldsymbol{w}}&=0\\
\frac{\partial L}{\partial b}&=0\\
\Rightarrow \boldsymbol{w}&=\sum\limits_{i=1}^m{\alpha_iy_i\boldsymbol{x}_i}\\
0&=\sum\limits_{i=1}^m\alpha_iy_i.\\
\\
\text{plug in:}\\
\max\limits_{\alpha}&\ \sum\limits_{i=1}^m{\alpha_i}-\frac{1}{2}\sum\limits_{i=1}^m\sum\limits_{j=1}^m\alpha_i\alpha_jy_iy_j\boldsymbol{x_i^Tx_j}\\
\text{s.t.}& \ \sum\limits_{i=1}^m \alpha_iy_i=0\\
&\ \alpha_i \geq0,i=1,2,...m

\end{align}
$$

> use **SMO** to compute $\boldsymbol{\alpha}$

****

## Kernel function

​	if such a plane doesn't exist, we can map the sample space into a higher dimensional space called **feature space**:
$$
\boldsymbol{x_i} \ \mapsto \ \phi{(\boldsymbol{x_i})}
$$
​	and we repeat the process above but introduce a new method called **kernel trick** when it comes to the computation of the inner product of $\phi(\boldsymbol{x_i}^T)\phi(\boldsymbol{x_j})$ 
$$
\kappa (\boldsymbol{x_i,x_j}) \stackrel{def}{=}\phi(\boldsymbol{x_i}^T)\phi(\boldsymbol{x_j})
$$

> **kernel matrix** is semi-positive $\Leftrightarrow$ symmetric function $\boldsymbol{\kappa(\cdot,\cdot)}$ is a **kernel function**
> $$
> \mathbf{K}\stackrel{def}{=}\{\kappa(\boldsymbol{x_i},\boldsymbol{x_j})\}_{m\times m},\ i,j=1,2,...,m.
> $$

****

## soft margin

​	some mistakes are allow to make, thus we need a **loss function** to measure the committed mistakes. and therefore the target we are going to minimize changes as below: 
$$
\min\limits_{\boldsymbol{w},b}\ \frac{1}{2}||\boldsymbol{w}^2||+C\sum\limits_{i=1}^m\mathcal{l}_{loss}(y_i(\boldsymbol{w^	Tx_i}+b)-1)\\
$$

> 1. we allow to make some mistakes to avoid **overfitting**
> 2. different **loss function** impacts the performance
> 3. the first item describes **the width of the margin**, the second item is about the **mistakes made on training set.**

**regularization problem:**
$$
\min\limits_{f} \Omega(f)+C\sum\limits_{i=1}^ml_{loss}(f(\boldsymbol{x_i},y_i))
$$

> 1. $\Omega(f)$: **structural risk** (最主要的优化对象)
> 2. $\sum\limits_{i=1}^ml_{loss}(f(\boldsymbol{x_i},y_i))$: **empirical risk** （在训练集上犯错的度量）

****

# Chapter 7 Bayesian decision making

## basic algorithm

​	suppose we have $N$ labels: $\mathcal{Y}=\{c_1,c_2,...c_N\}$, $\lambda_{ij}$ is the loss of mistaking $c_j$ for $c_i$

​	$P(c_i|\boldsymbol{x})$ is the probability of sample $\boldsymbol{x}$ belonging to $c_i$.

​	$R(c_i|\boldsymbol{x})=\sum\limits_{j=1}^N\lambda_{ij}P(c_j|\boldsymbol{x})$ is the loss of categorizing $\boldsymbol{x}$ into $c_i$. 

​	we are to find a function $h:\mathcal{X} \mapsto \mathcal{Y}$ to minimize:
$$
R(h)=\mathbb{E}_\boldsymbol{x}[R(h(\boldsymbol{x})|\boldsymbol{x})].
$$
​	which is equivalent to minimizing:
$$
R(h(\boldsymbol{x})|\boldsymbol{x})
$$
​	so the mapping $h$ maps sample $\boldsymbol{x}$ to the label that minimizing the conditional loss $R(c|\boldsymbol{x})$ :
$$
h^*(\boldsymbol{x}) = arg\min\limits_{c\in \mathcal{Y}} R(c|\boldsymbol{x})
$$

****

​	the thing is that how do we figure out the posterior probability $P(c_i|\boldsymbol{x})$

**Method 1: discriminative models**

​	decision tree, BP neuron network, support vector machine

**Method 2 : generative models**
$$
P(c|\boldsymbol{x})=\frac{P(\boldsymbol{x},c)}{P(\boldsymbol{x})}
$$

> ​	$P(c)$ is the prior probability, which can be estimated by the proportion of label $c$  in the sample space.
>
> ​	$P(\boldsymbol{x}|c)$ is the class-conditional probability, or likelihood, which means the probability of observing this sample given the label $c$.
>
> ​	$P(\boldsymbol{x})$ is the evidence factor, which is irrelevant to label $c$.
>
> ​	**so the whole thing is to estimate $P(\boldsymbol{x}|c)$**

​	$P(\boldsymbol{x}|c)$ cannot be estimated according to the frequency, because usually a great number of sample are not observed due to the limitation of the sample size, which is why we need to implement the **maximum likelihood estimation.**

​	the reason why it is hard to estimate is that $P(\boldsymbol{x}|c)$ is the joint pdf of all attributes, which is usually not well-demonstrated under the limitation of sample size.



****

​	==**the process of training a probability model is the process of parameter estimation**==	

****

​	we assume that $P(\boldsymbol{x}|c)$ belongs to a family of distribution with unknown parameter $\boldsymbol{\theta}_c$, and denote  $P(\boldsymbol{x}|c)$ as  $P(\boldsymbol{x}|\boldsymbol{\theta}_c)$.

​	$D_c$ is a collection of samples labeled with $c$.
$$
\begin{align}
P(D_c|\boldsymbol{\theta}_c)&=\prod\limits_{\boldsymbol{x} \in D_c} P(\boldsymbol{x}|\boldsymbol{\theta}_c)\\
LL(\theta_c)&=\log{P(D_c|\boldsymbol{\theta}_c)}\\
&=\sum\limits_{\boldsymbol{x}\in D_c}\log{P(\boldsymbol{x}|\boldsymbol{\theta}_c)}
\\
\boldsymbol{\hat{\theta}}_c&=arg\max\limits_{\boldsymbol{\theta}_c}{LL(\boldsymbol{\theta}_c)}\\


\end{align}
$$

****

## naive bayes classifier

​	here proposes **attribute conditional independence assumption**:
$$
P(c|\boldsymbol{x})=\frac{P(\boldsymbol{x},c)}{P(\boldsymbol{x})}=\frac{P(c)}{P(\boldsymbol{x})}\prod\limits_{i=1}^dP(x_i|c)
$$
​	based on this assumption, we can estimate $P(x_i|c)$ by:
$$
\begin{align}
\text{In discrete case:}\ P(x_i|c)&=\frac{|D_{c,x_i}|}{|D_c|}\\
\text{In continuous case:}\ P(x_i|c)&\sim \mathcal{N}(\mu_{c,i},\sigma_{c,i}^2)\\


\end{align}
$$
​	sometimes this method can be problematic, if the sample size is so small that some attribute values are not observed, leading some conditional probability to be wrongly **zero**, which means we need to implement **Laplacian correction**
$$
\hat{P}(x_i|c)=\frac{|D_{c,x_i}|+1}{|D_c|+N_i}
$$

> $N_i$ is the number of all the possible values of attribute $i$.

****

## Semi-naive Bayes classifiers

​	we don't assume that all attributes are independent, but believe that some attributes have high reliance with some unique attribute.
$$
P(c|\boldsymbol{x}) \varpropto P(c)\prod\limits_{i=1}^dP(x_i|c,pa_i)
$$
​	$pa_i$ is the **parent attribute** of attribute $i$

****

## E-M algorithm

> E-M algorithm is used to estimate the **latent variable**

​	suppose some variables are missing, which means we cannot observe some attribute value of some samples. 

​	$\mathbf{X} $ is a set of observed variables.

​	$\mathbf{Z} $ is a set of latent variables.

​	$\mathbf{\Theta}$ is the parameters of the model.

​	suppose we are going to do the MLE of   $\mathbf{\Theta}$, which means we need to maximize the log-likelihood:
$$
LL(\Theta|\mathbf{X,Z})=\ln{P(\mathbf{X,Z}|\Theta)}
$$
​	however, since $\mathbf{Z}$ is unknown, we cannot do the MLE directly, but do MLE on the **marginal likelihood**
$$
LL(\Theta|\mathbf{X})=\ln{P(\mathbf{X}|\Theta)}=\ln{\mathbb{E}_\mathbf{Z}[P(\mathbf{X,Z}|\Theta)|\mathbf{Z}]}
$$
​	**E_M algorithm** is a iterative method, at first we compute the expectation of the latent variable $\mathbf{Z}^1$ based on some initial parameter $\Theta^0$ and then we do the MLE of $\Theta$ and renew it to be $\Theta^1$, and go on do the iteration.
$$
\begin{align}
\text{start with\ }&\Theta^0\\
\Theta^n &\mapsto \mathbf{Z}^n\\
\mathbf{Z}&\mapsto\Theta^{n+1}\\

\end{align}
$$

> 用$\Theta$ 推断 $\mathbf{Z}$, 然后用$\mathbf{Z}$去推断$\Theta $



# Chapter 9 clustering

​	$D=\{\boldsymbol{x_1,x_2,...x_m}\}$ is a sample space without labels.

​	$\boldsymbol{x_i}=(x_{i1},x_{i2},x_{i3},...x_{in})$ is a n-dimensional vector, which means an instance doesn't have labels, but only attributes.

​	by clustering, $D$ is divided into several disjoint clusters $\{C_l|l=1,2,...,k\}$

​	$\lambda_j\in\{1,2,...k\}$ is the cluster label of sample $\boldsymbol{x_j}$

​	$\boldsymbol{\lambda}=(\lambda_1;\lambda_2;...;\lambda_m)$ is clustering outcome.

****

## validity index

​	high in intra-clustering similarity

​	low in inter-clustering similarity

**Method 1: external index**

​	we compare the training clustering model with the reference model to measure the clustering performance.
$$
\begin{align}
\mathcal{C}&=\{C_1,C_2,...,C_k\} \text{is the training clusters.}\\
\mathcal{C}^*&=\{C_1^*,C_2^*,...,C_k^*\} \text{is the reference clusters.}\\
\\
a&\stackrel{def}{=}|SS|,SS=\{(\boldsymbol{x_i,x_j})|\lambda_i=\lambda_j,\lambda_i^*=\lambda_j^*,i<j\}\\
b&\stackrel{def}{=}|SD|,SS=\{(\boldsymbol{x_i,x_j})|\lambda_i=\lambda_j,\lambda_i^*\neq\lambda_j^*,i<j\}\\
c&\stackrel{def}{=}|DS|,SS=\{(\boldsymbol{x_i,x_j})|\lambda_i\neq\lambda_j,\lambda_i^*=\lambda_j^*,i<j\}\\
d&\stackrel{def}{=}|SS|,SS=\{(\boldsymbol{x_i,x_j})|\lambda_i\neq\lambda_j,\lambda_i^*\neq\lambda_j^*,i<j\}
\\
&\text{larger a, the better; less d, the better.}\\
\\
&\textbf{here by we give the external index as below:}\\
&\textbf{Jaacard Coefficient:}\\
JC&=\frac{a}{a+b+c}\\
&\textbf{FM index}\\
FMI&=\sqrt{\frac{a}{a+b}\frac{a}{a+c}}\\
&\textbf{Rand index}\\
RI&=\frac{2(a+d)}{m(m-1)}\\
\end{align}
$$
**Method 2 : internal index**

​	we measure the clustering performance by some internal index:
$$
\begin{align}
\text{avg}(C)&=\frac{\sum\limits_{1\leq i<j\leq |C|}\text{dist}(\boldsymbol{x_i,x_j})}{\frac{|C|(|C|-1)}{2}},\ \text{the average distance of samples within the same cluster}\\
\text{diam}(C)&=\max\limits_{1\leq i<j\leq |C|}\text{dist}(\boldsymbol{x_i,x_j}), \ \text{the distance between the two furthest points within a cluster}\\
d_{min}(C_i,C_j)&=\min\limits_{\boldsymbol{x_i}\in C_i.\boldsymbol{x_j}\in C_j}\text{dist}(\boldsymbol{x_i,x_j}),\ \text{the distance between two clusters}\\
\boldsymbol{\mu_i}&\stackrel{def}{=}\frac{\sum\limits_{1\leq i\leq|C|}\boldsymbol{x_i}}{|C_i|},\ \text{the center of a cluster}\\
d_{cen}(C_i,C_j)&=\text{dist}(\boldsymbol{\mu_i,\mu_j}),\ \text{the distance between two centers}\\
\\
&\textbf{here by we give the internal index as below:}\\
&\textbf{DB index}\\
\text{DBI}&=\frac{1}{k}\sum\limits_{i=1}^k\max\limits_{j\neq i}(\frac{\text{avg}(C_i)+\text{avg}(C_j)}{d_{cen}(\boldsymbol{\mu_i,\mu_j})}),\ \text{the smaller, the distance within a cluster is smaller}\\
&\textbf{Dunn index}\\
\text{DI}&=\min\limits_{1\leq i\leq k}\{\min\limits_{j\neq i }(\frac{d_{min}(C_i,C_j)}{\max\limits_{1\leq l\leq k} \text{diam}(C_l)})\},\text{the larger, the distance between clusters larger}\\





\end{align}
$$

> numerical attribute = continuous attribute
>
> nominal attribute = categorical attribute

**VDM: a method for non-ordinal attribute**

​	$m_{u,a}$ the sum of samples taking value $a$ in attribute $u$.

​	$m_{u,a,i}$ the sum of samples in $i$th cluster taking value $a$ in attribute $u$.
$$
\text{VDM}_p(a,b)=\sum\limits_{i=1}^k|\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}|^p
$$

****

## K-means clustering

​	k-means clustering is a kind of **prototype-based clustering**.

> the most typical sample point is a prototype vector

$$
\begin{align}
E&=\sum\limits_{i=1}^k\sum\limits_{\boldsymbol{x}\in C_i}||\boldsymbol{x-\mu_i}||_2^2\\
\boldsymbol{\mu_i}&=\frac{1}{|C_i|}\sum\limits_{\boldsymbol{x}\in C_i}\boldsymbol{x}\\
\end{align}
$$

​	the procedure of k-means algorithm is:

> 1. figure out how many possible clusters.
>
> 2. suppose we have figure out the clusters number $k$, and then we randomly pick $k$ samples to be the **initial mean vector** $\{\boldsymbol{\mu_1,...,\mu_k}\}$.
>
> 3. for every sample $\boldsymbol{x_j},j=1,...,m$ in the dataset, we calculate the distance between $\boldsymbol{x_j}$ and every mean vector $\boldsymbol{\mu_i}$ to find out the closest mean vector to determine which cluster should sample $\boldsymbol{x_j}$ belongs to.
>    $$
>    \lambda_j=arg\min\limits_{i=1,2,...,k}||\boldsymbol{x_j-\mu_i}||_2
>    $$
>    
>
> 4. renew the **mean vector** $\boldsymbol{\mu_i}$ to $\boldsymbol{\mu_i'}$.
>    $$
>    \boldsymbol{\mu_i'}=\frac{1}{|C_i|}\sum\limits_{\boldsymbol{x}\in C_i} \boldsymbol{x}
>    $$
>
> 5. repeat step 3 and 4, until the **mean vector** is stable.

****

## Learning Vector Quantization

​	LVQ is also a prototype-based clustering, but LVQ is supervised, which means the samples $\boldsymbol{x}_i$ have known label $y_i$.

​	$D=\{(\boldsymbol{x_i},y_i)\}_{i=1}^m,\ y_j\in\mathcal{Y}$ . the algorithm of LVQ is to learn a set of prototype vectors $\{\boldsymbol{p_1,...,p_q}\}$. suppose $\mathcal{Y}=\{t_1,...,t_k\}$

> 1. we randomly pick a sample $\boldsymbol{x_i}$ from all the sample labeled with $t_q$, to be the original prototype vector $\boldsymbol{p_q}$. By doing so, we obtain a set of original prototype vector $\{\boldsymbol{p_1,...,p_q}\}$.
>
> 2. for every sample $\boldsymbol{x_j}$, we calculate the distance between $\boldsymbol{x_j}$ and $\boldsymbol{p_i},i=1,...,q$, and figure out the closet prototype vector $\boldsymbol{p_{i^*}}$
>    $$
>    i^*=arg \min\limits_{i\in \{1,2,...,q\}}||\boldsymbol{x_j-p_i}||_2
>    $$
>
> 3. **if  $\boldsymbol{x_j}$ and $\boldsymbol{p_{i^*}}$ have the same label, then pull $\boldsymbol{p_{i^*}}$ closer to $\boldsymbol{x_j}$, otherwise push $\boldsymbol{p_{i^*}}$ away from $\boldsymbol{x_j}$:**
>    $$
>    \left\{ \begin{array}{cc}\boldsymbol{p'}=\boldsymbol{p_{i^*}}+\eta\cdot(\boldsymbol{x_j-\boldsymbol{p_{i^*}}}),&\ \text{if} \ y_j=t_{i^*}\\
>    \boldsymbol{p'}=\boldsymbol{p_{i^*}}-\eta\cdot(\boldsymbol{x_j-\boldsymbol{p_{i^*}}}),&\ \text{if} \ y_j\neq t_{i^*}\\
>    \end{array} \right.
>    $$
>
> 4. renew $\boldsymbol{p_{i^*}}$ to be $\boldsymbol{p'}$.

****

## Mixture of Gaussian clustering

​	suppose we have a dataset $D$ whose samples generate a sample space $\mathcal{X}$:
$$
\begin{array}{ccccc}
\mathbf{X}&\mathbf{X_1}&\mathbf{X_2}&...&\mathbf{X_n}\\
\boldsymbol{x_1}&x_{11}&x_{12}&...&x_{1n}\\
\boldsymbol{x_2}&x_{21}&x_{22}&...&x_{2n}\\
.&.&.&.&.\\
\boldsymbol{x_j}&x_{j1}&x_{j2}&...&x_{jn}\\
.&.&.&.&.\\
\boldsymbol{x_m}&x_{m1}&x_{m2}&...&x_{mn}\\
\end{array}
$$
​	we believe that sample $\boldsymbol{x_j}$ comes from a mixed Gaussian distribution"

>(we know in advance that the mixture has $k$ components)

$$
\begin{align}
\mathbf{X}&\sim \sum\limits_{i=1}^k \alpha_i \mathcal{N}(\boldsymbol{\mu_i,\Sigma_i})\\
p_{\mathcal{M}}(\boldsymbol{x_j})&=\sum\limits_{i=1}^k \alpha _ip(\boldsymbol{x_j}|\boldsymbol{\mu_i,\Sigma_i})\\
\\

\sum\limits_{i=1}^k \alpha_i &= 1, \ \text{$\alpha_i$\ is mixture coefficient}\\
p(\boldsymbol{x_j}|\boldsymbol{\mu_i,\Sigma_i})&=\frac{1}{(2\pi)^{\frac{n}{2}}|\boldsymbol{\Sigma_i}|^{\frac{1}{2}}}e^{-\frac{1}{2}\boldsymbol{(x_j-\mu_i)^T\Sigma_i^{-1}(x_j-\mu_i)}}\\

\end{align}
$$

​	"$z_j=i$" denotes the event that "$\boldsymbol{x_j}$ comes from the $i$th component of the mixture Gaussian",then:
$$
\begin{align}
p_{\mathcal{M}}(z_j=i|\boldsymbol{x_j})&=\frac{P(z_j=i)\cdot p_\mathcal{M}(\boldsymbol{x_j}|z_j=i)}{p_{\mathcal{M}}(\boldsymbol{x_j})}\\
&=\frac{\alpha_i\cdot p_\mathcal{M}(\boldsymbol{x_j|\mu_i,\Sigma_i})}{\sum\limits_{l=1}^kp_{\mathcal{M}}(\boldsymbol{x_j|\mu_l,\Sigma_l})}\\
\\
\gamma_{ji}&\stackrel{def}{=}p_{\mathcal{M}}(z_j=i|\boldsymbol{x_j})\\
\lambda_j&=arg\max\limits_{i\in\{1,2,...,k\}}\gamma_{ji}\\
\end{align}
$$

> 1. $p_{\mathcal{M}}(z_j=i|\boldsymbol{x_j})$ is the posterior of $\boldsymbol{x_j}$ being generated by the $i$th component.
>
> 2. we determine which cluster sample $\boldsymbol{x_j}$ should belong to by figuring out the maximum of the posterior probability.
>
> 3. use EM- algorithm:
>    $$
>    \begin{align}
>    \gamma_{ji}&\mapsto \{(\alpha_i,\boldsymbol{\mu_i,\Sigma_i})|1\leq i\leq k\}\\
>    \{(\alpha_i,\boldsymbol{\mu_i,\Sigma_i})|1\leq i\leq k\}&\mapsto \gamma_{ji}
>    \end{align}
>    $$

​	**EM algorithm**:

> 1. initialize $\{(\alpha_i,\boldsymbol{\mu_i,\Sigma_i})|1\leq i\leq k\}$ and then calculate $\gamma_{ji}$ by:
>    $$
>    \gamma_{ji}=p_{\mathcal{M}}(z_j=i|\boldsymbol{x_j})=\frac{\alpha_i\cdot p_\mathcal{M}(\boldsymbol{x_j|\mu_i,\Sigma_i})}{\sum\limits_{l=1}^kp_{\mathcal{M}}(\boldsymbol{x_j|\mu_l,\Sigma_l})}
>    $$
>
> 2. since we have $\gamma_{ji}$, we figure the MLE $\{(\hat{\alpha_i},\boldsymbol{\hat{\mu_i},\hat{\Sigma}_i})|1\leq i\leq k\}$
>    $$
>    \begin{align}
>    LL(D)&=\ln{\prod\limits_{j=1}^mp_{\mathcal{M}}(\boldsymbol{x_j})}\\
>    &=\sum\limits_{j=1}^m\ln{\sum\limits_{i=1}^k\alpha_i\cdot p(\boldsymbol{x_j|\mu_i,\Sigma_i})}\\
>    \frac{\partial LL(D)}{\partial \boldsymbol{\mu_i}}&=\sum\limits_{j=1}^m\frac{\alpha_i\cdot p(\boldsymbol{x_j|\mu_i,\Sigma_i})}{\sum\limits_{l=1}^k\alpha_l\cdot p(\boldsymbol{x_j|\mu_l,\Sigma_l})}(\boldsymbol{x_j-\mu_i})=0\\
>    \Rightarrow \boldsymbol{\mu_i}&=\frac{\sum\limits_{j=1}^m\gamma_{ji}\boldsymbol{x_j}}{\sum\limits_{j=1}^m\gamma_{ji}}\\
>    \frac{\partial LL(D)}{\partial \boldsymbol{\Sigma_i}}&=0\\
>    \Rightarrow\boldsymbol{\Sigma_i}&=\frac{\sum\limits_{j=1}^m\gamma_{ji}(\boldsymbol{x_j-\mu_i})(\boldsymbol{x_j-\mu_i})^T}{\sum\limits_{j=1}^m\gamma_{ji}}\\
>       
>    &\text{to get the estimation of $\alpha_i$ we use the Lagrange multipliers}\\
>    &\left\{\begin{array}{c}
>    LL(D)=\sum\limits_{j=1}^m\ln{\sum\limits_{i=1}^k\alpha_i\cdot p(\boldsymbol{x_j|\mu_i,\Sigma_i})}\\
>    \sum\limits_{i=1}^k \alpha_i=1\\
>    \end{array}\right.\\
>    \Rightarrow f(\boldsymbol{\alpha})&=LL(D)+\lambda(\sum\limits_{i=1}^k\alpha_i-1)\\
>    \frac{\partial f(\boldsymbol{\alpha})}{\partial \boldsymbol{\alpha}}&=\sum\limits_{j=1}^m\frac{p(\boldsymbol{x_j|\mu_i,\Sigma_i})}{\sum\limits_{l=1}^k\alpha_l\cdot p(\boldsymbol{x_j|\mu_l,\Sigma_l})}+\lambda=0\\
>    &...\\
>    \alpha_i&=\frac{1}{m}\sum\limits_{j=1}^m\gamma_{ji}\\
>    \end{align}
>    $$

****

**Density-based clustering**

**Hierarchical clustering**

​	**AGNES** method do the clustering from the bottom to the top or from the top to the bottom, the essence of this algorithm is to decide which **distance function ** to implement:
$$
\begin{align}
d_{min}(C_i,C_j)&=\min\limits_{\boldsymbol{x\in C_i},\boldsymbol{y}\in C_j} \text{dist}(\boldsymbol{x,y})\\
d_{max}(C_i,C_j)&=\max\limits_{\boldsymbol{x\in C_i},\boldsymbol{y}\in C_j} \text{dist}(\boldsymbol{x,y})\\
d_{avg}(C_i,C_j)&=\frac{1}{|C_i||C_j|}\sum\limits_{\boldsymbol{x}\in C_i}\sum\limits_{\boldsymbol{y}\in C_j} \text{dist}(\boldsymbol{x,y})\\

\end{align}
$$
​	the algorithm of hierarchical clustering:

> 1. calculate all the distances between any given two clusters and figure out the minimum.
>    $$
>    (C_{i}^*,C_{j}^*)=arg\min\limits_{C_i,C_j}d(C_i,C_j),\ \forall \ i,j=1,...,m\\
>    $$
>
> 2. combine $C_i^*$ and  $C_j^*$, and go on.
> 3. repeat......

# Chapter 10 reduction of dimensinality

**kNN is a method of lazy learning**

> 1. **1NN(最近邻分类器)** 的错误率小于贝叶斯最佳分类器错误率的两倍



> curse of dimensionality

## Methods of dimension reduction:

**Method 1: Multiple Dimensional Scaling(MDS)**

​	the basic mindset is to map the original sample space $\mathcal{X}$ into some low-dimension space $\mathcal{Z}$ under some constriants.

​	before starting the deduction, we need to make some agreement:

> 1. every instance is a **row vector**.


$$
\begin{align}
\mathcal{X}&= \left(\begin{array}{c}
\boldsymbol{x_1}\\
\boldsymbol{x_2}\\
.\\
.\\
.\\
\boldsymbol{x_m}\\
\end{array}\right)
= \left(\begin{array}{c}
x_{11}&x_{12}&...&x_{1d}\\
x_{21}&x_{22}&...&x_{2d}\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
x_{m1}&x_{m2}&...&x_{md}\\
\end{array}\right)_{m\times d}\\
D_{\mathcal{X}}&=\left(\begin{array}{c}
\text{dist}_{11}&\text{dist}_{12}&...&\text{dist}_{1d}\\
\text{dist}_{21}&\text{dist}_{22}&...&\text{dist}_{2d}\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
\text{dist}_{m1}&\text{dist}_{m2}&...&\text{dist}_{md}\\
\end{array}\right)_{m\times m}\\
\text{dist}_{ij}&=\text{dist}(\boldsymbol{x_i,x_j})\\
\\

\end{align}
$$
​	we want to transform $\mathcal{X}$ into $\mathcal{Z}$, with $D_{\mathcal{X}} =D_{\mathcal{Z}}$
$$
\begin{align}
\mathcal{Z}&= \left(\begin{array}{c}
\boldsymbol{z_1}\\
\boldsymbol{z_2}\\
.\\
.\\
.\\
\boldsymbol{z_m}\\
\end{array}\right)
= \left(\begin{array}{c}
z_{11}&z_{12}&...&z_{1d'}\\
z_{21}&z_{22}&...&z_{2d'}\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
z_{m1}&z_{m2}&...&z_{md'}\\
\end{array}\right)_{m\times d'}\\
\\

D_{\mathcal{Z}}&=\left(\begin{array}{c}
||\boldsymbol{z_1-z_1}||&||\boldsymbol{z_1-z_2}||&...&||\boldsymbol{z_1-z_{d'}}||\\
||\boldsymbol{z_2-z_1}||&||\boldsymbol{z_2-z_2}||&...&||\boldsymbol{z_2-z_{d'}}||\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
||\boldsymbol{z_m-z_1}||&||\boldsymbol{z_m-z_2}||&...&||\boldsymbol{z_m-z_{d'}}||\\
\end{array}\right)_{m\times m}\\

\\
B&\stackrel{def}{=}ZZ^T=\left(\begin{array}{c}
\boldsymbol{z_1}\\
\boldsymbol{z_2}\\
.\\
.\\
.\\
\boldsymbol{z_m}\\
\end{array}\right)\boldsymbol{(z_1^T,z_2^T,...,z_m^T)}\\
&=\left(\begin{array}{c}
\boldsymbol{z_1z_1^T}&\boldsymbol{z_1z_2^T}&...&\boldsymbol{z_1z_m^T}\\
\boldsymbol{z_2z_1^T}&\boldsymbol{z_2z_2^T}&...&\boldsymbol{z_2z_m^T}\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
\boldsymbol{z_mz_1^T}&\boldsymbol{z_mz_2^T}&...&\boldsymbol{z_mz_m^T}\\
\end{array}\right)_{m\times m}\\
&=\left(\begin{array}{c}
b_{11}&b_{12}&...&b_{1m}\\
b_{21}&b_{22}&...&b_{2m}\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
b_{m1}&b_{m2}&...&b_{mm}\\
\end{array}\right)_{m\times m}

\end{align}
$$
​	here we use $D_{Z}$ to calculate $B$:
$$
\begin{align}
\text{dist}_{ij}^2&=||\boldsymbol{z_i-z_j}||^2=\boldsymbol{||z_i||^2+||z_j||^2-2z_iz_j^T}=b_{ii}+b_{jj}-2b_{ij}\\
\sum\limits_{i=1}^m\text{dist}_{ij}&=tr(B)+m\cdot b_{jj}\\
\sum\limits_{j=1}^m\text{dist}_{ij}&=tr(B)+m\cdot b_{ii}\\
\sum\limits_{j=1}^m\sum\limits_{i=1}^m\text{dist}_{ij}^2&=2m\cdot tr(B)\\
b_{ij}&=\frac{1}{2}(\frac{1}{m}\sum\limits_{i=1}^m\text{dist}_{ij}^2+\frac{1}{m}\sum\limits_{j=1}^m\text{dist}_{ij}^2-\text{dist}_{ij}^2-\frac{1}{m^2}\sum\limits_{j=1}^m\sum\limits_{i=1}^m\text{dist}_{ij}^2)
\end{align}
$$
​	we do orthogonal similarity diagonalization to $B$

> 1. $B$ is a semi-positive 

$$
\begin{align}
&\exists P:P^T=P^{-1},P=(P_1,P_2,...,P_m)\\
B&=(P_1,P_2,...,P_m)\left(\begin{array}{c}
\lambda_1&0&...&0\\
0&\lambda_2&...&0\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
0&0&...&\lambda_m\\
\end{array}\right)\left(\begin{array}{c}
P_1^T\\
P_2^T\\
.\\
.\\
.\\
P_m^T\\
\end{array}\right), \lambda_1\geq\lambda_2\geq...\geq\lambda_m\\ 
&\text{suppose}\ \lambda_{d^*+1}=...=\lambda_{m}=0\\
\Rightarrow B&=(P_1,P_2,...,P_{{d^*}})\left(\begin{array}{c}
\lambda_1&0&...&0\\
0&\lambda_2&...&0\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
0&0&...&\lambda_{d^*}\\
\end{array}\right)\left(\begin{array}{c}
P_1^T\\
P_2^T\\
.\\
.\\
.\\
P_{d^*}^T\\
\end{array}\right)\\
&=(P_1,P_2,...,P_{{d^*}})\left(\begin{array}{c}
\sqrt{\lambda_1}&0&...&0\\
0&\sqrt{\lambda_2}&...&0\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
0&0&...&\sqrt{\lambda_{d}}
\end{array}\right)\left(\begin{array}{c}
\sqrt{\lambda_1}&0&...&0\\
0&\sqrt{\lambda_2}&...&0\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
0&0&...&\sqrt{\lambda_{d}}
\end{array}\right)\left(\begin{array}{c}
P_1^T\\
P_2^T\\
.\\
.\\
.\\
P_{d^*}^T\\
\end{array}\right)\\
\Rightarrow Z&=(P_1,P_2,...,P_{{d^*}})\left(\begin{array}{c}
\sqrt{\lambda_1}&0&...&0\\
0&\sqrt{\lambda_2}&...&0\\
.&.&...&.\\
..&.&...&.\\
.&.&...&.\\
0&0&...&\sqrt{\lambda_{d}}
\end{array}\right)

\end{align}
$$

​	thus we figure out the new sample space $\mathcal{Z}$



**Method 2: Principal Component Analysis(PCA)**

​	$\boldsymbol{x_j},j=1,...,m$ is a d-dimension **column vector**.

​	$\boldsymbol{w_i},i=1,...,d'(d'<d)$ is a d-dimension **column vector**.

​	$||\boldsymbol{w_i}||^2=1,(\boldsymbol{w_i,w_j})=\boldsymbol{w_i}^T\boldsymbol{w_j}=0,\forall\ i\neq j.$

​	$W=(\boldsymbol{w_1,w_2,...,w_{d}})_{d\times d}$.

​	$X=(\boldsymbol{x_1,x_2,...,x_m})_{d\times m}$

​	the mindset is that: $W$ is a ultra plane we are going to figure out, on which we can map the original sample $\boldsymbol{x_j}$ to become $\boldsymbol{y_j}=\sum\limits_{i=1}^{d'}(\boldsymbol{x_j,w_i})=W^T\boldsymbol{x_j}$

>  we denote $\boldsymbol{\alpha_j}$ is a abstract vector and $\boldsymbol{x_j}$ is the coordinate of it.
> $$
> \begin{align}
> \boldsymbol{\alpha_j}&=(\boldsymbol{e_1,e_2,...,e_d})\left(\begin{array}{c}
> x_{j1}\\
> x_{j2}\\
> .\\
> .\\
> .\\
> x_{jd}\\
> \end{array}\right)\\
> &=(\boldsymbol{w_1,w_2,...,w_{d}})\left(\begin{array}{c}
> \boldsymbol{w_1}^T\\
> \boldsymbol{w_2}^T\\
> .\\
> .\\
> .\\
> \boldsymbol{w_d}^T\\
> \end{array}\right)\left(\begin{array}{c}
> x_{j1}\\
> x_{j2}\\
> .\\
> .\\
> .\\
> x_{jd}\\
> \end{array}\right)\\
> \\
> \Rightarrow \boldsymbol{y_j}&=W^T\boldsymbol{x_j}\\
> 
> 
> \end{align}
> $$
>  

​	the optimal target is to maximize the variance of the new sample $\boldsymbol{y_j}$.
$$
\begin{align}
Var&=\frac{1}{m}(\sum\limits_{j=1}^m\boldsymbol{y_j\boldsymbol{y_j}^T})\\
&=\frac{1}{m}\sum\limits_{j=1}^mW^T\boldsymbol{x_j}(W^T\boldsymbol{x_j})^T\\
&=\frac{1}{m}W^T(\sum\limits_{j=1}^m\boldsymbol{x_j}\boldsymbol{x_j}^T)W
\\
\\
f(W)&=tr(W^TXX^TW)\\
W^TW&=I\\
l(W)&=??(这里怎么用拉格朗日乘子法)\\
\\
XX^TW&=\lambda W\\
...&\text{(do orthogonal similarity diagonalization to } XX^T)
\end{align}
$$


**Method 3: Kernelized PCA(KPCA)**

​	non-linear method for PCA.

****

**metric learning:**

​	we learn a **metric matrix** the measure the distance between samples.
$$
\text{dist}_{math}^2(\boldsymbol{x_i,x_j})=(\boldsymbol{x_i-x_j})^T\mathbf{M}(\boldsymbol{x_i-x_j})
$$

# Chapter 11 feature selection and sparse learning

>  relevant feature: valuable
>
> irrelevant feature: useless
>
> redundant feature: can be deduced by other features

## Methods of feature selection

**Method 1: Relief features**

​	$(\boldsymbol{x_i},y_i)$ is an instance

​	$\boldsymbol{x}_{i,nh}$ near hit (最近的同类样本)

​	$\boldsymbol{x}_{i,nm}$ near miss (最近的异类样本)
$$
\delta^j=\sum\limits_{i=1}^m-\text{diff}(x_i^j,x_{i,nh}^j)^2+\text{diff}(x_i^j,x_{i,nm}^j)^2
$$

> $\delta^j$是分给属性j的权，如果最近的同类样本的距离比最近的异类样本的距离要小，说明这个属性在“把这个示例从正确归类”这一预测行为上是有益的，应该加大权重。

**Method 2: wrapper **

​	LVW(Las Vegas method)

​	time-consuming

**Method 3: embedding**

​	we do regularization under $L_1$ norm which is embedding.
$$
\min\limits_{\boldsymbol{w}}\sum\limits_{i=1}^m(y_i-\boldsymbol{w}^T\boldsymbol{x_i})^2+\lambda||\boldsymbol{w}||_1
$$
​	we implement **proximal gradient descent(PGD)** to compute $\boldsymbol{w}$
$$
\begin{align}
&\min\limits_{\boldsymbol{x}}f(\boldsymbol{x})+\lambda||\boldsymbol{x}||_1\\
\hat{f}(\boldsymbol{x})&\simeq f(\boldsymbol{x}_k)+\langle\nabla f(\boldsymbol{x_k}),\boldsymbol{x-x_k}\rangle+\frac{L}{2}||\boldsymbol{x-x_k}||^2\\
&=\frac{L}{2}||\boldsymbol{x}-(\boldsymbol{x_k}-\frac{1}{L}\nabla f(\boldsymbol{x_k})||_2^2 + const\\
\\
\boldsymbol{x}_{k+1}&=arg\min\limits_{\boldsymbol{x}}\frac{L}{2}||\boldsymbol{x}-(\boldsymbol{x_k}-\frac{1}{L}\nabla f(\boldsymbol{x_k})||_2^2+\lambda||\boldsymbol{x}||_1\\
...\\
\end{align}
$$

****

## Code learning and Sparse representation

​	we are trying to figure out a dictionary, which is also a matrix,  and sample $\boldsymbol{x_i}$ will be **more sparse** if reconstructed under such matrix.

​	$\{\boldsymbol{x_1,x_2,...,x_m}\}$ is a data set

​	$\mathbf{B}_{d\times k}$ is a dictionary we are going to learn. $k$ is the vocabulary of the dictionary.

​	$\boldsymbol{\alpha_i}$ is the **sparse representation** of $\boldsymbol{x_i}$
$$
\min\limits_{\mathbf{B},\boldsymbol{\alpha_i}}\sum\limits_{i=1}^m||\boldsymbol{x_i}-\mathbf{B}\boldsymbol{\alpha_i}||_2^2+\lambda\sum\limits_{i=1}^m||\boldsymbol{\alpha_i}||_1
$$
​	we want to figure out $\mathbf{B}$ and $\boldsymbol{\alpha_i}$

​	step 1: fix $\mathbf{B}$, calculate $\boldsymbol{\alpha_i}$ by PGD.

​	step 2: fix $\boldsymbol{\alpha_i}$, renew  $\mathbf{B}$ by minimizing:
$$
\min\limits_{\mathbf{B}} ||\mathbf{X}-\mathbf{BA}||_F^2\\
$$

> $\mathbf{A}_{k\times m}=\{\boldsymbol{\alpha_1,\alpha_2,...,\alpha_m}\}$
>
> $\mathbf{X}_{d\times m }=\{\boldsymbol{x_1,x_2,...,x_m}\}$







































