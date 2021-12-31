---
title: Fitting a Poisson mixture model using EM.
subtitle: In this Blogpost we will derive the equations required to fit Poisson mixture from scratch and implement the model using Python.
layout: default
date: 2021-12-31
keywords: blogging, writing
published: true
---

Count data are pervasive in science, particularly in biology where hightroughput sequencing experiments create huge datasets comprising for example counts genes. Here, however we will consider a synthetic dataset which we will create using `numpy` and `scipy`. The following code block creates a three component Poisson mixture.

{% highlight python %}
N = 500
π = np.array([.3, .4, .3])
μ = np.array([30, 100, 150])
Z = stats.multinomial(p=π, n=1).rvs(N) 
_, c = np.where(Z==1)
X = stats.poisson(μ[c]).rvs()  
{% endhighlight %}

To get a better feeling for these data it instructive to quickly look at the resulting histogram of $X$.

![Cloudflare architecture](/assets/2021-12-31-poisson_mixture/histogram.png){: .center-image }

The plot nicely illustrates the three clusters in the data centered at roughly 30, 100 and 150. Now, imagine that you'd been given these data and your task is to determine the proportion of data belonging to each of the three clusters and their respective cluster centers.

## Defining the model

Let us now formalise the problem by introducing all necessary variables. We denote our (synthetic) data with  {% katexmm %}$\mathbf{X}=\{x_1, \dots, x_N\}$  {% endkatexmm %}. Now to define the Poisson mixture model 

{% katexmm %}
$$ x_n|\mu_k \sim \text{Pois}(x_n; \mu_k)$$
{% endkatexmm %}


## The EM algorithm

Now before we derive the equations to infer the mixture proportions and means, let us quickly recap the EM algorithm {% cite bishop2006pattern %}.

1. Initialise the parameters  {% katexmm %} $\boldsymbol{\theta}$ {% endkatexmm %}
2. **E-step**: Compute the posterior distribution {% katexmm %} $p(\mathbf{Z}|\mathbf{X};\theta^{\text{old}})$ {% endkatexmm %}.
3. **M-step**: Determine the parameter updates by minimising the expected joint log likelihood under the posterior {% katexmm %} $\boldsymbol{\theta}^{\text{new}}=\argmin_{\theta}\mathbb{E}_{\mathbf{Z}\sim p(\mathbf{Z}|\mathbf{X};\theta^{\text{old}})}[p(\mathbf{Z},\mathbf{X};\theta^{\text{old}})]$ {% endkatexmm %}.





{% katexmm %}
$$
e = mc^2. \tag{1}
$$
{% endkatexmm %}

Cool!