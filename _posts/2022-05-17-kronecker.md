---
title: Kronecker Matrix products
subtitle: In this blogpost I'll review some of the properties that come along with Kronecker products.
layout: default
date: 2022-05-17
keywords: blogging, writing
published: false
---

Kroecker products do not often arise on a day to day basis, hence the more surprsing are their properties. In the following we let capital letter $\mathbf{A},\mathbf{B}$ and $\mathbf{C}$ be matrices, $\mathbf{a},\mathbf{b}$ and $\mathbf{c}$ vectors and greek letter $\alpha,\beta$ denote scalars.

## Definition

We begin with the definition of a Kroecker product. The Kronecker product $\mathbf{A}\otimes\mathbf{B}$ where $\mathbf{A}\in\mathbb{R}^{m\times n}$ and $\mathbf{B}\in\mathbb{R}^{p\times q}$ is a $mp\times nq$ matrix $\mathbf{C}$. To illustrate this let us consider

{% katexmm %}

$$
\mathbf{a}=\left[\begin{array}{l}0 \\ 1\end{array}\right] \quad B=\left[\begin{array}{ll}2 & 3 \\ 4 & 5\end{array}\right]
$$

{% endkatexmm %}
