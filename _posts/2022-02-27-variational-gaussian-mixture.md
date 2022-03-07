---
title: Revisiting Variational Inference for Statististican
subtitle: Variational Inference - A Review for Statisticians is perhaps the go to paper in order to learn variational inference (VI). After all, the paper has over 2800 citations indicating its popularity in the community. I recently decided to reread the paper while trying to closely follow the derivations. In this blogpost, I'll extend the derivations of the Gaussian Mixture model of the paper in the hope to elucidate some of the steps over which the authors went quickly.
layout: default
date: 2022-02-27
keywords: blogging, writing
published: true
---

Blei et. al. illustrate the coordinate ascent variational inference (CAVI) using a simple Gaussian Mixture model {% cite blei2017varinf %}. The model[^1] places a prior on the mean of each component while keeping the variance of the likelihood fixed.

{% katexmm %}
$$
\begin{aligned} 
\mu_{k} & \sim \mathcal{N}\left(0, \sigma^{2}\right) \\ 
\mathbf{z}_{n} & \sim \text { Categorical }(1 / K, \dots, 1 / K) \\ 
x_{n} \mid \mathbf{z}_{n}, \boldsymbol{\mu} & \sim \mathcal{N}\left(\mathbf{z}_{n}^{\top}\boldsymbol{\mu}, 1\right) 
\end{aligned}
$$
{% endkatexmm %}

In the following, we will derive the joint probability and CAVI update equations for the model. Finally, we use these equations to implement the model in Python. 

## Constructing the log joint

We start by defining the components of the model. Note that we can write the probability of the prior component means as 

{% katexmm %}
$$
p(\boldsymbol{\mu})=\prod_k \mathcal{N}(\mu_k|0, \sigma^2).
$$
{% endkatexmm %}

Similarly, the prior for the latent variables $\mathbf{z}_n$ may be expressed as

{% katexmm %}
$$
p(\mathbf{z}_{n})=\prod_k \left(\frac{1}{K}\right)^{z_{nk}}
$$
{% endkatexmm %}

while the likelihood is given by

{% katexmm %}
$$
p(x_n|\boldsymbol{\mu}, \mathbf{z}_{n})=\prod_k \mathcal{N}(0|\mu_k, 1)^{z_{nk}}.
$$
{% endkatexmm %}

We now introduce the variables {% katexmm %} $\mathbf{X} = \{x_n\}_{n=1}^{N}${% endkatexmm %} and {% katexmm %}$\mathbf{Z}=\{ \mathbf{z}_n\}_{n=1}^{N}${% endkatexmm %} to denote the complete dataset. Note that $p(\mathbf{Z})$ and $p(\mathbf{X}\|\boldsymbol{\mu}, \mathbf{Z})$ are simply

{% katexmm %}
$$
p(\mathbf{Z})=\prod_n\prod_k \left(\frac{1}{K}\right)^{z_{nk}}\quad\text{and}\quad p(\mathbf{X}|\boldsymbol{\mu}, \mathbf{Z})=\prod_n \prod_k \mathcal{N}(0|\mu_k, 1)^{z_{nk}}.
$$
{% endkatexmm %}

With these equations we can construct the joint distribution which factorizes as follows

{% katexmm %}
$$
p(\mathbf{X}, \boldsymbol{\mu}, \mathbf{Z})= p(\boldsymbol{\mu}) p(\mathbf{X}|\boldsymbol{\mu}, \mathbf{Z}) p(\mathbf{Z})= \prod_k \mathcal{N}(\mu_k|0, \sigma^2) \prod_n\prod_k \left(\frac{1}{K}\cdot \mathcal{N}(0|\mu_k, 1)\right)^{z_{nk}}.
$$
{% endkatexmm %}

Finally, we end up with the following log joint distribution for the model

{% katexmm %}
$$
\log{p(\mathbf{X}, \boldsymbol{\mu}, \mathbf{Z})} = \sum_k \log{\mathcal{N}(\mu_k|0, \sigma^2)} +\sum_n\sum_k z_{nk} \left(\log{\frac{1}{K}}+ \log{\mathcal{N}(0|\mu_k, 1)}\right).\tag{1}
$$
{% endkatexmm %}

## The variational density for the mixture assignments

To obtain the (log) variational distribution of $\mathbf{z}_n$, we simply take the expectation of the log joint $(1)$ with respect to all other variables of the model. In our simple Gaussian mixture model this corresponds to $q(\mu_k)$, as it is the only other variable of the model.

{% katexmm %}
$$
 \begin{aligned} 
 \log q^{*}\left(\mathbf{z}_{n}\right) &=\mathbb{E}_{q(\mu_k)}[\log p(x_n, \boldsymbol{\mu}, \mathbf{z}_n)] +\text { const. } \\ 
 &=\mathbb{E}_{q(\mu_k)}\left[\log p\left(x_{n} | \boldsymbol{\mu}, \mathbf{z}_{n}\right)+\log p\left(\mathbf{z}_{n}\right)\right]+\text { const. } \\ 
 &=\mathbb{E}_{q(\mu_k)}\left[\sum_{k} z_{nk}\left(\log \frac{1}{K}+\log \mathcal{N}\left(0 \mid \mu_{k}, 1\right)\right)\right]+\operatorname{const.}  \\ 
 &=\mathbb{E}_{q(\mu_k)}\left[-\cancel{\sum_{k} z_{n k} \log \frac{1}{K}}+\sum_{k} z_{n k}\left(-\frac{1}{2} \log 2 \pi-\frac{1}{2}\left(x_{n}-\mu_{k}\right)^{2}\right)\right] +\operatorname{const.} \\
 &=\mathbb{E}_{q(\mu_k)}\left[-\cancel{\sum_{k}  \frac{z_{n k}}{2} \log 2 \pi} -\sum_{k}  \frac{z_{n k}}{2}\left(x_{n}^2-2x_n\mu_k+\mu_{k}^2\right)\right] +\operatorname{const.} \\
 &=\mathbb{E}_{q(\mu_k)}\left[-\sum_{k}  \cancel{\frac{z_{n k}}{2} x_{n}^2} - z_{n k} x_n\mu_k+ \frac{z_{n k}}{2} \mu_{k}^2\right] +\operatorname{const.} \\
 &=\sum_{k} z_{n k} x_n\mathbb{E}_{q(\mu_k)}[\mu_k] - \frac{z_{n k}}{2} \mathbb{E}_{q(\mu_k)}[\mu_{k}^2] +\operatorname{const.} \\
 &=\sum_{k} z_{n k} \left(x_n\mathbb{E}_{q(\mu_k)}[\mu_k] - \frac{1}{2} \mathbb{E}_{q(\mu_k)}[\mu_{k}^2]\right) +\operatorname{const.} \\
 &=\sum_{k} z_{n k} \log{\rho_{nk}} +\operatorname{const.} \tag{2}
 \end{aligned} 
$$
{% endkatexmm %}

Here I have canceled constant terms in $z_{nk}$ (only terms including the expectations w.r.t. to $q(\mu_k)$ change). Let's take a closer look at the last line of $(2)$; exponentiating reveals $\log q^{*}(\mathbf{z}_n)$ that it has the form of a multinomial distribution

{% katexmm %}
$$ 
q^{*}\left(\mathbf{z}_{n}\right)\propto \prod_{k} \rho_{nk} ^ {z_{n k}},
$$
{% endkatexmm %}

thus in order to normalise the distribution, we require that the variational parameter $\rho_{nk}$ represents a probability. We therefore define 

{% katexmm %}
$$ 
r_{nk} = \frac{\rho_{nk}}{\sum_j \rho_{nj}} = \frac{e^{x_n\mathbb{E}_{q(\mu_k)}[\mu_k] - \frac{1}{2} \mathbb{E}_{q(\mu_k)}[\mu_{k}^2]}}{\sum_j e^{x_n\mathbb{E}_{q(\mu_j)}[\mu_j] - \frac{1}{2} \mathbb{E}_{q(\mu_j)}[\mu_{j}^2]}}
$$
{% endkatexmm %}

and the our final density is given by

{% katexmm %}
$$ 
q^{*}\left(\mathbf{z}_{n};\mathbf{r}_n\right) = \prod_{k} r_{nk} ^ {z_{n k}}.\tag{3}
$$
{% endkatexmm %}

## The variational density for the means

We proceed similarly to determine the variational density of $q(\mu_k)$

{% katexmm %}
$$
 \begin{aligned} 
 \log q^{*}\left(\mathbf{\mu}_{k}\right) &=\mathbb{E}_{q(\mathbf{z}_n)}[\log p(\mathbf{X}, \boldsymbol{\mu}, \mathbf{Z})] +\text { const. } \\ 
 &=\mathbb{E}_{q(\mathbf{z}_n)}\left[\log p\left(\boldsymbol{\mu}\right) + \log p\left(\mathbf{X} | \boldsymbol{\mu}, \mathbf{Z}\right)\right]+\text { const. } \\ 
 &=\mathbb{E}_{q(\mathbf{z}_n)}\left[\log{\mathcal{N}(\mu_k|0, \sigma^2)}+\sum_{n} z_{nk} \log \mathcal{N}\left(0 \mid \mu_{k}, 1\right)\right]+\operatorname{const.}  \\ 
 &=\mathbb{E}_{q(\mathbf{z}_n)}\left[-\cancel{\frac{1}{2}\log{2\pi\sigma^2}}-\frac{1}{2\sigma^2}\mu_k^2+ \sum_{n} z_{n k}\left(\cancel{-\frac{1}{2} \log 2 \pi}-\frac{1}{2}\left(x_{n}-\mu_{k}\right)^{2}\right)\right] +\operatorname{const.} \\
 &=\mathbb{E}_{q(\mathbf{z}_n)}\left[-\frac{1}{2\sigma^2}\mu_k^2 -\sum_{n}  \frac{z_{n k}}{2}\left(x_{n}^2-2x_n\mu_k+\mu_{k}^2\right)\right] +\operatorname{const.} \\
 &=-\frac{1}{2\sigma^2}\mu_k^2 +\mathbb{E}_{q(\mathbf{z}_n)}\left[-  \cancel{\sum_{n}\frac{z_{n k}}{2} x_{n}^2} + \mu_k\sum_{n} z_{n k} x_n - \mu_{k}^2\sum_{n}\frac{z_{n k}}{2} \right] +\operatorname{const.} \\
 &=-\frac{1}{2\sigma^2}\mu_k^2 + \mu_k\sum_{n} \mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n - \mu_{k}^2\sum_{n}\frac{\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]}{2} +\operatorname{const.} \\
 &= \mu_k\sum_{n} \mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n - \mu_{k}^2(\sum_{n}\frac{\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]}{2}+\frac{1}{2\sigma^2}) +\operatorname{const.} \\
 &=\begin{bmatrix} \mu_k \\ \mu_k^2 \end{bmatrix}^T\begin{bmatrix} \mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n \\ -(\frac{1}{2}\sum_{n}\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]+\frac{1}{\sigma^2}) \end{bmatrix} +\operatorname{const.}
 \end{aligned} 
$$
{% endkatexmm %}

The last line of the derivation suggests that the variational distribution for $\mu_k$ is Gaussian with natural parameter {% katexmm %}$\boldsymbol{\eta}=[\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n, -(\sum_{n}\frac{\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]}{2}+\frac{1}{2\sigma^2})]${% endkatexmm %} and sufficient statistic {% katexmm %}$t(\mu_k)=[\mu_k, \mu_k^2]${% endkatexmm %}. Using standard formulas {% cite blei2016exponential %}, we find that the mean posterior mean and covariance are given by

{% katexmm %}
$$ 
s^2_k=-\frac{1}{2\eta_2}=\frac{1}{\sum_{n}\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]+\frac{1}{\sigma^2}}\quad\text{and}\quad m_k=\eta_1\cdot s_k^2=\frac{\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}] x_n}{\sum_{n}\mathbb{E}_{q(\mathbf{z}_n)}[z_{n k}]+\frac{1}{\sigma^2}}.\tag{4}
$$
{% endkatexmm %}

## Solving expectations

Although we have derived parameters of our variational distributions, we can't work properly with the results as all of them contain unresolved expectations. However, we can leverage the form of our variational distributions, i.e. $z_{nk}$ and $\mu_k$ are respectively multinomial and normally distributed. For example, to solve the expectation of $z_{nk}$, we use $(3)$ to determine

{% katexmm %}
$$ 
\mathbb{E}_{q_(\mathbf{z}_n)}[z_{nk}]=\sum_{\mathbf{z}}\mathbf{z}_n q^{*}(\mathbf{z}_n; r_n)=\sum_{\mathbf{z}}\mathbf{z}_n \prod_{k} r_{nk} ^ {z_{n k}} = r_{nk}.\tag{5}
$$
{% endkatexmm %}

Now we can simply plug $(5)$ into $(4)$ to obtain

{% katexmm %}
$$ 
\sigma^2_N=\frac{1}{\sum_{n}r_{nk}+\frac{1}{\sigma^2}}\quad\text{and}\quad\mu_N=\frac{r_{nk} x_n}{\sum_{n}r_{nk}+\frac{1}{\sigma^2}}.
$$
{% endkatexmm %}

It is easy to see that {% katexmm %}$\mathbb{E}_{q(\mu_k)}[\mu_k]=m_k${% endkatexmm %}. To determine the second moment of $\mu_k$, which is also required to compute $r_{nk}$, we make use of standard properties of the variance[^2]

{% katexmm %}
$$ 
\mathbb{E}_{q(\mu_k)}[\mu_k^2]=m_k^2+s_k^2.
$$
{% endkatexmm %}

## Implementing the model

With these equation in hand we can easily implement the model.


{% highlight python %}
class GaussianMixtureCavi:
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.m = np.random.uniform(np.min(X), np.max(X), K)
        self.s = np.random.normal(size=K) ** 2
        self.σ = 1

    def fit(self):
        for it in range(100):
            y = self.X.reshape(-1, 1) * self.m.reshape(1, -1) - (
                0.5 * (self.s + self.m**2)
            ).reshape(1, -1)
            α = np.max(y, 1).reshape(-1, 1)
            self.ϕ = np.exp(y - (α + np.log(np.exp(y - α).sum(1, keepdims=True))))
            denom = 1 / self.σ + self.ϕ.sum(0, keepdims=True)
            self.m = (self.ϕ * self.X.reshape(-1, 1)).sum(0) / denom
            self.s = 1 / denom

    def approx_mixture(self, x):
        return np.stack(
            [
                ϕ_i * stats.norm(loc=m_i, scale=1).pdf(x)
                for m_i, ϕ_i in zip(self.m.squeeze(), self.ϕ.mean(0).squeeze())
            ]
        ).sum(0)
{% endhighlight %}

The following plot illustrates a fit of the model to simulated data with $N=100$, $\mu=[-4, 0, 9]$ and equal mixture component probabilities.

![CAVI Gaussian mixture model fit.](/assets/2022-02-27-variational_gaussian_mixture/fit.png){: .center-image }







[^1]: Note that I have slightly altered the notation of the paper using $\mathbf{z}$ instead of $\mathbf{c}$ and $n$ instead of $i$.
[^2]: {% katexmm %}$\operatorname{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2$ {% endkatexmm %}

<!-- Count data are ubiquitous in science. For example, in biology hightroughput sequencing experiments create huge datasets of gene counts. In this blogpost, I will take a closer look at the Expectation Maximisation (EM) algorithm and use it to derive a Poisson mixture model. To get started, however, we will simulate some data from a Poisson mixture using `numpy` and `scipy`. 

{% highlight python %}
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

N = 500
π = np.array([.3, .4, .3])
μ = np.array([30, 100, 150])
Z = stats.multinomial(p=π, n=1).rvs(N) 
_, c = np.where(Z==1)
X = stats.poisson(μ[c]).rvs()  
{% endhighlight %}

To get a better feeling for these data it is instructive to quickly plot a histogram.

{% highlight python %}
for i in range(len(π)):
    _ = plt.hist(X[c==i], bins=np.arange(200), label=f'Component {i+1}')

plt.legend()
plt.xlabel('$X$')
plt.ylabel('Counts')
sns.despine()
plt.savefig('histogram.png', bbox_inches='tight', pad_inches=0)
{% endhighlight %}

![Simulated data](/assets/2021-12-31-poisson_mixture/histogram.png){: .center-image }

The plot nicely illustrates the three clusters in the data centered at roughly 30, 100 and 150. Now, imagine that you had been given these data and your task is to determine the proportion of data belonging to each of the three clusters and their respective cluster centers. In the following, we will first motivate the usage of the EM algorithm, apply it to our problem and finally implement the model in `Python`.

## Defining a Poisson mixture

Let us now formalise the problem. We denote our (synthetic) data with  {% katexmm %}$\mathbf{X}=\{x_1, \dots, x_N\}$  {% endkatexmm %} and let {% katexmm %} $\boldsymbol{\mu}=\{\mu_1,\dots,\mu_K\}${% endkatexmm %} be the vector representing the cluster means of cluster {% katexmm %} $k=\{1,\dots, K\}$ {% endkatexmm %}. If we are given the cluster $k$ for a data point $x_n$ we can compute its likelihood using

{% katexmm %}
$$
\begin{aligned} 
p(x_n|k;\boldsymbol{\mu})= \text{Pois}(x_n; \mu_k).
\end{aligned} 
$$
{% endkatexmm %}

A finite mixture of such Poisson distributions can be expressed as

{% katexmm %}
$$
\begin{aligned} 
p(x_n;\boldsymbol{\mu}, \boldsymbol{\pi})= \sum_{k=1}^{K}\pi_k \text{Pois}(x_n; \mu_k)
\end{aligned} 
$$
{% endkatexmm %}

and the likelihood for the whole dataset $\mathbf{X}$ is given by

{% katexmm %}
$$
p(\mathbf{X};\boldsymbol{\mu}, \boldsymbol{\pi}) = \prod_{n=1}^{N} \sum_{k=1}^{K} \pi_k \text{Pois}(x_n;\mu_k). \tag{1}
$$
{% endkatexmm %}

From $(1)$ we can see that it is difficult to optimise the log-likelihood

{% katexmm %}
$$
\log{p(\mathbf{X};\boldsymbol{\mu}, \boldsymbol{\pi})} = \sum_{n=1}^{N} \log{\sum_{k=1}^{K} \pi_k \text{Pois}(x_n;\mu_k)}, \tag{2}
$$
{% endkatexmm %}

as this expression involves the log of sum, making it hard to find close form solutions for the parameters of the model. 

## The EM algorithm

In the previous section we found that it is infeasible to optimise the marginal log likelihood (Eq. $(2)$). In such cases we can employ the EM algorithm to simplify the optimisation problem. In particular, we will introduce for each data point $x_n$ a corresponding latent variable $\mathbf{z}_n$, and derive the log joint distribution of $x_n$ and {% katexmm %} $\mathbf{z}_n$ {% endkatexmm %}, which is easier to optimise than Eq. $(2)$. 

Before we derive the equations to infer the parameters {% katexmm %} $\boldsymbol{\theta}=\{\boldsymbol{\mu},\boldsymbol{\pi} \}$ {% endkatexmm %} of our Poisson mixture, let us quickly recap the EM algorithm {% cite bishop2006pattern %}.

1. Initialise the parameters of the model {% katexmm %} $\boldsymbol{\theta}^{\text{old}}$ {% endkatexmm %}.
2. **E-step**: Compute the posterior distribution of the latent variable {% katexmm %} $p(z_{nk}|x_n;\boldsymbol{\theta}^{\text{old}})$ {% endkatexmm %} using the current parameter estimates.
3. **M-step**: Determine the parameter updates by minimising the expected joint log likelihood under the posterior determined in the E-Step
    {% katexmm %}
    $$
    \boldsymbol{\theta}^{\text{new}}=\argmin_{\theta}\mathbb{E}_{z_{nk}\sim p(z_{nk}|x_n;\boldsymbol{\theta}^{\text{old}})}[\log{p(\mathbf{Z},\mathbf{X};\boldsymbol{\theta})}].
    $$
    {% endkatexmm %}
4. Check for convergence by means of the log likelihood or parameter values. If the convergence criterion is not satisfied update  

    {% katexmm %}
    $$
    \boldsymbol{\theta}^{\text{old}}\leftarrow\boldsymbol{\theta}^{\text{new}}
    $$
    {% endkatexmm %}

    and return to step 2.

### The joint log likelihood

The first step involves to determine the joint log likelihood of the model. In finite mixture models, {% katexmm %} $\mathbf{z}_n=(z_{n1}, \dots, z_{nK})^T$ {% endkatexmm %} is a binary $K$-dimensional vector in which a single component equals $1$. Hence, we can write the conditional distribution of $x_n$ given $\mathbf{z}_n$

{% katexmm %}
$$
p(x_n|\mathbf{z}_n;\boldsymbol{\mu})= \prod_{k=1}^{K} \text{Pois}(x_n; \mu_k)^{z_{nk}}
$$
{% endkatexmm %}

and the prior for {% katexmm %} $\mathbf{z}_n$ {% endkatexmm %} as

{% katexmm %}
$$
 p(\mathbf{z}_n; \boldsymbol{\pi})=\prod_{k=1}^{K} \pi_{k}^{z_{nk}}.
$$
{% endkatexmm %}

The joint probability distribution (for the complete data) is therefore

{% katexmm %}
$$
 p(\mathbf{X},\mathbf{Z};\boldsymbol{\theta})=\prod_{n=1}^{N} p(x_n|\mathbf{z}_n;\boldsymbol{\mu}) \cdot p(\mathbf{z}_n; \boldsymbol{\pi})=\prod_{n=1}^{N}\prod_{k=1}^{K}(\pi_k\text{Pois}(x_n; \mu_k))^{z_{nk}}
$$
{% endkatexmm %}

and the log joint 

{% katexmm %}
$$
\log{p(\mathbf{X},\mathbf{Z};\boldsymbol{\theta})}=\sum_{n=1}^{N}\sum_{k=1}^{K}z_{nk}\{\log{\pi_k}+\log{\text{Pois}(x_n; \mu_k)}\}.\tag{3}
$$
{% endkatexmm %}

Comparing Eq. $(2)$ and $(3)$ reveals that in the the latter equation the logarithm distributes over the prior and the conditional distribution. This greatly simplifies the optimisation problem.

### Setting up the E-step: The latent posterior distribution

To find the posterior distribution of $z_{nk}$ we make use of Bayes rule

{% katexmm %}
$$
p(z_{nk}|x_n;\boldsymbol{\theta}^{\text{old}})=\frac{p(z_{nk},x_n;\boldsymbol{\theta}^{\text{old}})}{p(x_n;\boldsymbol{\theta}^{\text{old}})}=\frac{\prod_{k=1}^{K}(\pi^{\text{old}}_k\text{Pois}(x_n; \mu^{\text{old}}_k))^{z_{nk}}}{\sum_z \prod_{j=1}^{K}(\pi^{\text{old}}_j\text{Pois}(x_n; \mu^{\text{old}}_j))^{z_{nj}}}
$$
{% endkatexmm %}

which can be simplified by noting $\mathbf{z}_n$ is a binary vector

{% katexmm %}
$$
p(z_{nk}|x_n;\boldsymbol{\theta}^{\text{old}})=\frac{\pi^{\text{old}}_k\text{Pois}(x_n; \mu^{\text{old}}_k)}{\sum_{j=1}^{K}\pi^{\text{old}}_j\text{Pois}(x_n; \mu^{\text{old}}_j)}.\tag{4}
$$
{% endkatexmm %}

### Completing the M-step

Now that we have derived the joint log likelihood and the posterior distribution of the latent variables, we can take its expectation

{% katexmm %}
$$
\mathbb{E}_{z_{nk}\sim p(z_{nk}|x_{n};\boldsymbol{\theta}^{\text{old}})}[\log{p(\mathbf{Z},\mathbf{X};\boldsymbol{\theta})}]=\sum_{n=1}^{N}\sum_{k=1}^{K} \mathbb{E}_{z_{nk}\sim p(z_{nk}|x_{n};\boldsymbol{\theta}^{\text{old}})}[z_{nk}]\{\log{\pi_k}+\log{\text{Pois}(x_n; \mu_k)}\}
$$
{% endkatexmm %}

which requires us to calculate

{% katexmm %}
$$
\mathbb{E}_{z_{nk}\sim p(z_{nk}|x_{n};\boldsymbol{\theta}^{\text{old}})}[z_{nk}]=\sum_{k=1}^K z_{nk} p(z_{nk}|x_n;\boldsymbol{\theta}^{\text{old}})=\frac{\pi^{\text{old}}_k\text{Pois}(x_n; \mu^{\text{old}}_k)}{\sum_{j=1}^{K}\pi^{\text{old}}_j\text{Pois}(x_n; \mu^{\text{old}}_j)}=\gamma(z_{nk}).
$$
{% endkatexmm %}

Thus we obtain

{% katexmm %}
$$
\mathbb{E}_{z_{nk}\sim p(z_{nk}|x_{n};\boldsymbol{\theta}^{\text{old}})}[\log{p(\mathbf{Z},\mathbf{X};\boldsymbol{\theta})}]=\sum_{n=1}^{N}\sum_{k=1}^{K} \gamma(z_{nk}) \{\log{\pi_k}+\log{\text{Pois}(x_n; \mu_k)}\}.\tag{5}
$$
{% endkatexmm %}

Finally, we need to compute the derivatives with respect to the model parameters and set them to zero

{% katexmm %}
$$
\frac{\partial \mathbb{E}_{z_{nk}\sim p(z_{nk}|x_{n};\boldsymbol{\theta}^{\text{old}})}[\log{p(\mathbf{Z},\mathbf{X};\boldsymbol{\theta})}]}{\partial \mu_k}=0\quad\text{and}\quad\frac{\partial \mathbb{E}_{z_{nk}\sim p(z_{nk}|x_{n};\boldsymbol{\theta}^{\text{old}})}[\log{p(\mathbf{Z},\mathbf{X};\boldsymbol{\theta})}]}{\partial \pi_k}=0.
$$
{% endkatexmm %}

Solving for $\mu_k$ and $\pi_k$ respectively gives us the following update rules 

{% katexmm %}
$$
\mu_k=\frac{\sum_{n=1}^{N}\gamma(z_{nk})x_n}{\sum_{n=1}^{N}\gamma(z_{nk})}\quad\text{and}\quad\pi_k=\frac{\sum_{n=1}^{N}\gamma(z_{nk})}{N}.\tag{6}
$$
{% endkatexmm %}

Note that we have to use Lagrange Multipliers in order to obtain the update rule for $\pi_k$.

## Implementing the model

After having done the hard math, the actual implementation of the model is straight forward. The `PoissonMixture` class takes the as arguments the number of desired clusters $K$ and allows to specifify the initial parameters in the  `__init__` method. The implementation of the E-Step and and M-Step directly follows from Eq. $(4)$ and $(6)$. Finally, the `PoissonMixture` model provides a function to compute negative log likelihood[^1] of the model as well as a `fit` method that takes the count data as input.

{% highlight python %}
class PoissonMixture():
    def __init__(self, K=2, π_init=None, μ_init=None, max_iter=10):
        self.K = K
        self.max_iter= max_iter
        
        # initialise parameters
        self.μ_old = (
            np.random.choice(X.squeeze(), K).reshape(1, -1) 
            if μ_init is None else μ_init)
        self.π_old = (
            np.array([1/K for _ in range(K)]).reshape(1, -1) 
            if π_init is None else π_init)
        
    def e_step(self, X):
        γ = stats.poisson(self.μ_old).pmf(X) * self.π_old
        γ /= γ.sum(1, keepdims=True)
        return γ
    
    def m_step(self, X, γ):
        μ_new = (γ * X).sum(0) / γ.sum(0)
        π_new = γ.sum(0) / X.shape[0]
        return μ_new, π_new
    
    def nll(self, X, γ):
        return -(γ * (
            stats.poisson(self.μ_old).logpmf(X) 
            + np.log(self.π_old))).sum()
    
    def fit(self, X):
        
        self.history = {
            'nll': np.zeros(self.max_iter),
            'μ': np.zeros((self.max_iter, self.K)),
            'π': np.zeros((self.max_iter, self.K))
        }
        
        prev_nll = np.inf
        for step in range(self.max_iter):
            γ = self.e_step(X)
            μ_new, π_new = self.m_step(X, γ)
            curr_nll = self.nll(X, γ)
        
            self.history['nll'][step] = curr_nll
            self.history['μ'][step] = self.μ_old
            self.history['π'][step] = self.π_old
            
            Δ_nll = curr_nll - prev_nll 
            print(f'Step {i}: NLL={curr_nll:.2f}, Δ={Δ_nll:.2f}')
            prev_nll = curr_nll
            self.μ_old = μ_new
            self.π_old = π_new
{% endhighlight %}

To test our model we instatiate it provide the date via the `fit` method.

{% highlight python %}
m0 = PoissonMixture(3)
m0.fit(X)
Step 0: NLL=12339.65, Δ=-inf
Step 1: NLL=5619.75, Δ=-6719.90
Step 2: NLL=5249.60, Δ=-370.15
Step 3: NLL=4801.28, Δ=-448.32
Step 4: NLL=3458.16, Δ=-1343.11
Step 5: NLL=2372.49, Δ=-1085.67
Step 6: NLL=2346.74, Δ=-25.75
Step 7: NLL=2345.87, Δ=-0.87
Step 8: NLL=2345.85, Δ=-0.02
Step 9: NLL=2345.85, Δ=0.00
{% endhighlight %}

We find that the EM algorithm quickly converges to the optimal solutions, although obviously this implementation is for demonstration purposes only.

![Convergence of the model after 10 iterations.](/assets/2021-12-31-poisson_mixture/convergence.png){: .center-image }

## Conclusion

In this article we have derived the EM algorithm to fit Poisson mixture models. Further extensions of the model could involve the incorporation of a prior for $\boldsymbol{\mu}$ or $\boldsymbol{\pi}$.

[^1]: Technically the `nll` method computes the _expected joint log likelihood_ (Eq. $(5)$).

<!--- 
[^1]: Some footnote.
## Appendix

where  {% katexmm %} $\boldsymbol{\pi}=\{\pi_1, \dots, \pi_K\}$ {% endkatexmm %} represents a set of mixture weights, which are probabilities that sum to one. 

{% katexmm %}
$$
\begin{aligned} 
x_n|k & \sim \text{Pois}(x_n| k; \boldsymbol{\mu}) \\
k &\sim p(k)
\end{aligned} \tag{1}
$$
{% endkatexmm %}

where $p(k)=\pi_k$ simply represents a set of mixture weights, which are probabilities that sum to one. From $(1)$ we can deduce the marginal distribution of $x$ by marginalising $k$ over the joint distribution

{% katexmm %}
$$
p(x_n) = \sum_{k=1}^{K} p(x_n|k) = \sum_{k=1}^{K} p(k)p(x_n | k) = \sum_{k=1}^{K} \pi_k \text{Pois}(x_n;k).
$$
{% endkatexmm %}
-->
