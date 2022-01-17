---
title: Fully bayesian analysis of rat tumors
subtitle: In this Blogpost we will derive the equations required to fit Poisson mixture from scratch and implement the model using Python.
layout: default
date: 2022-01-10
keywords: blogging, writing
published: false
---

Recently, I worked through a chapter in Gelman's classic Bayesian Data Analysis {% cite gelmanbda04 %} book. Here, Gelman describes the analysis of a dataset describing the incidence of tumours in rats from historical experiments. These data basically consist of the number of rats that eventually developed a tumour and the total number of rodents involved in the experiment. To make this more explicit, the data looks as follows.

```
Previous experiments (randomly selected points)
0/20    1/20    2/23    4/19    6/20    16/52 etc.
```

Since we only have these data and no further information about the data such batch labels, we may model the data with _exchangeable_ parameters, i.e. we consider the class of models that model each $y_n$ with an associated parameter $\theta_n$ and the likelihood $p(y_n\|\theta_n)$. The term exchangeable reflects our ignorance about any ordering or grouping that could be made with the parameters $\theta_n$ if there was more more information about the data.

An exchangeable distribution has each parameter $\theta_n$ drawn as an independent sample from a prior/population distribution governed by some unknown parameter vector $\phi$. In our case, we use a Beta distribtion $\text{Beta}(\theta_n\| \alpha, \beta)$ as the data is naturally modeled with an Binomial and hence the population distribution is conjugate. For such a simple model, we can employ a full Bayesian analysis, which according to Gelman involves to

1. Find the joint posterior density as the product of the hyperprior distribtion $p(\phi)$, the population distribution $p(\theta\|\phi)$ and the likelihood $p(y\|\theta)$,
2. Determine analytically the conditional posterior density of $\theta$ given the hyperparameters $\phi$,
3. Estimate $\phi$ by determining its marginal distribution $p(\phi\|y)$.

## The model

Determining the joint posterior is straight forward. We consider the following factorisation

{% katexmm %}
$$
\begin{aligned}
p(\theta, \alpha, \beta | y) &\propto \underbrace{p(\alpha, \beta)}_{\text{hyperprior}}\underbrace{p(\theta|\alpha,\beta)}_{\text{prior}}\underbrace{p(y|\theta)}_{\text{likelihood}} \\
&=p(\alpha, \beta)\prod_{n=1}^N \text{Beta}(\theta_n|\alpha,\beta)\prod_{n=1}^N\text{Binomial}(y_n|\theta_j) \\
&=p(\alpha, \beta)\prod_{n=1}^N \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha) \Gamma(\beta)} \theta_{n}^{\alpha-1}\left(1-\theta_{n}\right)^{\beta-1} \prod_{n=1}^{N} \theta_{n}^{y_{n}}\left(1-\theta_{n}\right)^{x_{n}-y_{n}}\\ 
\end{aligned}\tag{1}
$$

{% endkatexmm %}

Now recall that the basic probability rule $P(A\|B)=\frac{P(A,B)}{P(B)}$ as we will use it to determine $p(\theta\|\alpha,\beta,y)$

{% katexmm %}
$$
\begin{aligned}
p(\theta| \alpha, \beta,  y) &= \frac{p(\alpha, \beta, \theta |y)}{p(\alpha, \beta)} \\
&\propto \frac{p(\alpha, \beta)p(\theta|\alpha,\beta)p(y|\theta)}{p(\alpha, \beta)} \\
&=\prod_{n=1}^N \text{Beta}(\theta_n|\alpha,\beta)\prod_{n=1}^N\text{Binomial}(y_n|\theta_j) \\
&=\prod_{n=1}^N \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha) \Gamma(\beta)} \theta_{n}^{\alpha-1}\left(1-\theta_{n}\right)^{\beta-1} \prod_{n=1}^{N} \theta_{n}^{y_{n}}\left(1-\theta_{n}\right)^{x_{n}-y_{n}}\\ 
&=N \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha) \Gamma(\beta)}\prod_{n=1}^N\theta_{n}^{\alpha+y_{n}-1}\left(1-\theta_{n}\right)^{\beta+x_{n}-y_{n}-1}.\tag{2}
\end{aligned}
$$
{% endkatexmm %}

Notice that this equation corresponds to a $Beta(\theta_n\|\alpha+y_n, \beta+x_n-y_n)$ distribution, thus

{% katexmm %}
$$
p(\theta| \alpha, \beta,  y) =\prod_{n=1}^N\frac{\Gamma(\alpha+\beta+x_n)}{\Gamma(\alpha+y_n) \Gamma(\beta+x_n+y_n)}\theta_{n}^{\alpha+y_{n}-1}\left(1-\theta_{n}\right)^{\beta+x_{n}-y_{n}-1}.\tag{3}
$$
{% endkatexmm %}

Finally we determine the marginal $p(\alpha, \beta\|y)$ by making use of the conditional probability formula again.


{% katexmm %}
$$
 p(\alpha, \beta \mid y) \propto p(\alpha, \beta) \prod_{n=1}^{N} \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha) \Gamma(\beta)} \frac{\Gamma\left(\alpha+y_{n}\right) \Gamma\left(\beta+x_{n}-y_{n}\right)}{\Gamma\left(\alpha+\beta+x_{n}\right)}.
$$
{% endkatexmm %}

## Setting up a noninformative hyper prior distribution

To express our uncertainty about the tumour rates $\theta_n$, we are tempted to choose prior $p(\alpha,\beta)=\text{const}$ as such a distribution would not favor any particular value of $\theta_n$. However, Gelman first proposes to reparameterise in terms of $\text{logit}\frac{\alpha}{\alpha+\beta}=\log{\frac{\alpha}{\beta}}$ and $\log{\alpha + \beta}$ which correspond to the logit of the mean the logarithm of the sample size. 

The interpretation of $\log{\frac{\alpha}{\beta}}$ as the logit of the mean was not entirely evident on first sight. However, it becomes clear when one remembers that the expectation of a $\text{Beta}(\alpha, \beta)$ distribution is given by $\frac{\alpha}{\alpha+\beta}$ and may be interpreted as a probability as it falls into the interval $[0,1]$  ($\alpha,\beta>0$). The logit function equals to the logarithm of the odds ratio $\frac{p}{1-p}$ where $p$ is a probability. Therefore


{% katexmm %}
$$
\text{logit}\frac{\alpha}{\alpha+\beta}=\log{\frac{\frac{\alpha}{\alpha+\beta}}{1-\frac{\alpha}{\alpha+\beta}}}=\log{\frac{\alpha}{\beta}}.
$$
{% endkatexmm %}

To understand why $\log{\alpha + \beta}$ may be understood as the logarithm of the sample size, recall a simple Beta Binomial model. Here, $\alpha$ and $\beta$ can be interpreted as the number prior successes and failures respectively, so that $\alpha+\beta$ can be interpreted as the effective sample size.

## Changing the p

Before we continue, let us denote $\eta_1=\log{\frac{\alpha}{\beta}}$ and $\eta_2=\log{\alpha+\beta}$. Now we really require $p(\eta_1,\eta_2)\propto 1$, but unfortunately the we have the posterior $(4)$ only in terms of $\alpha, \beta$. Luckily we can transform the variables using the change of varibles "formula" for probability distributions

{% katexmm %}
$$
p(\alpha, \beta)=p(\eta_1, \eta_2) \left|\begin{array}{cc} \frac{\partial\eta_1}{d\alpha} & \frac{\partial\eta_2}{d\beta} \\ \frac{\partial\eta_1}{\partial\alpha} & \frac{\partial\eta_2}{\partial\beta}\end{array}\right| 
$$
{% endkatexmm %}

