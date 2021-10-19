---
title: "Introduction to type-2 fuzzy sets"
date: 2021-10-15
tags: [type_2_fuzzy, fuzzy, set]
draft: false
---
In a previous post, we have seen how we use a type-1 fuzzy set when we cannot determine the membership of an element as 0 or 1. Similarly, when the circumstances are so fuzzy that we have trouble determining the membership grade even as a crisp number in [0, 1], we use a type-2 fuzzy set.

This post will look at the basic concepts behind type-2 fuzzy sets and how we can represent a type-2 set using the python type-2 fuzzy library. We will base this discussion on "Type-2 Fuzzy Sets made Simple" by Robert John and Jerry Mendel, possibly the best paper to learn about type-2 fuzzy sets and logic.

### What is a type-2 fuzzy set?

Type-2 fuzzy logic is motivated by the premise that concepts have different meanings to different people. If we, for example, had to ask two different persons to plot their perception of the fuzzy set hot in the temperature universe of discourse and constrain them to use a trapezoidal set for their description, almost certainly we will get different definitions.

![Difference in type_1](/post/img/type_2_example_1.png)

A type-2 fuzzy set A, ( $ \tilde{A} $ ) can be 

![Difference in type_2](/post/img/type_2_set.jpg)

The membership function of a Type-2 Fuzzy Set and Interval Type-2 Fuzzy Set is three dimensional, with the 

- x-axis called the **primary variable**
- the y-axis called the **secondary variable** or secondary domain denoted by $u$  and
- the z-axis called the **membership function value** (or secondary membership function value or secondary grade) that is denoted  by $ \mu $.

A general type-2 fuzzy set, $\tilde{A}$ can also be expressed using the **vertical slice representation** of $\tilde{A}$ :

$$\tilde{A} = \int_{\forall x \in X} \left[ \int_{\forall u \in J_{x}} f(u) / u  \middle] \right/ x$$

Denoting $\int\int$ as the union over all admissible $x$ and $u$ for continuous
universes of discourse (for discrete universes of discourse use $\sum\sum$ instead), $\tilde{A}$ can also be expressed as:

$$\tilde{A}=\int_{x\in X}\int_{u\in J_{x}} \mu_{\tilde{A}}(x,u) / (x,u)$$

where $J_{x}\subseteq[0,1]$

Finally, there is the **wavy-slice representation** which is also known as the **embedded type-2 fuzzy set representation** or the **Mendel-John representation** where an embedded type-2 fuzzy set $\tilde{A}_{e}^{j}$ is defined as a type-2 fuzzy set that has only one primary membership at each $x_i$, also referred to as the wavy-slice. The wavy-slice representation is therefore given as:

$$\tilde{A}=\bigcup_{\forall j} \tilde{A}_{e}^{j}$$

#### Vertical Slice

Type-2 fuzzy sets are often decomposed in terms of pairs of domain values and secondary membership functions known as the vertical slice representation. A vertical slice is Type-1 fuzzy set $\mu_{\tilde{A}}(x=x',u)$ for $x\in X$ and $\forall u \in J_{x'}\subseteq[0,1]$, that is:

$$\mu_{\tilde{A}}(x=x',u)=\int_{u\in J_{x'}}f_{x'}(u) / u$$

where $0\leq f_{x'}(u)\leq 1$

#### Secondary Membership Function

A **secondary membership function** is a vertical slice of  $\mu_{ \tilde{A} }(x, u)$. It is $\mu_{ \tilde{A} }(x=x', u)$ for $x \in X$ and $\forall u \in J_x \subseteq [0, 1]$

$$\mu_{ \tilde{A} }(x=x' , u) \equiv \mu_{ \tilde{A} }(x') = \int_{u \in J_{x'}} f_{x'}(u)/u ; J_{x'}\subseteq[0,1]$$

where $0 \leq f_{x'}(u) \leq 1 $ 

\begin{figure}[ht]
	\centering
	\includegraphics[width=0.7\linewidth]{chapters/02_type2_fuzzy/img/img_2}
	\caption[Vertical Slice]{Vertical Slice at $x=x'$}
	\label{fig:02:img2}
\end{figure}

$$\tilde{A}=\int_{x \in X} \mu_{ \tilde{A} }(x) /x = \int_{x \in X} \left[  \int_{u\in J_{x}} f(u) / u \right]  /x$$

$J_{x} \subseteq [0,1]$

#### Primary Membership

The **domain** of a secondary membership function is called the **primary membership** of $x$. Hence in

$$\tilde{A}=\int_{x \in X} \mu_{ \tilde{A} }(x) /x = \int_{x \in X} \left[  \int_{u\in J_{x}} f(u) / u \right]  /x$$

$J_{x}$ is the primary membership function, where $J_{x} \subseteq [0,1]$ for $\forall x \in X$

#### Secondary Grade

The **amplitude** of a secondary membership function is the **secondary grade**. Hence in

$$\tilde{A}=\int_{x \in X} \mu_{ \tilde{A} }(x) /x = \int_{x \in X} \left[  \int_{u\in J_{x}} f(u) / u \right]  /x$$

where $J_{x} \subseteq [0,1]$, $f(u)$ is the secondary grade.

If $X$ and $J_{x}$ are discrete:

$$\tilde{A} = \sum_{x \in X} \left[ \sum_{u \in J_{x}} f(u) / u \right] /x$$

\begin{center}
\begin{tabular}{rl}
	$\tilde{A}$ & $ = \displaystyle \sum_{x \in X} \left[ \sum_{u \in J_{x}} f(u) / u \right] /x$ \\ 
	&  = $ \displaystyle \sum_{i=1}^{N} \left[  \sum_{u \in J_{x_{i}}} f_{x_{i}}(u) / u \right] / x_{i}$ \\  
	&  = $ \displaystyle \sum_{k=1}^{M_{1}} \left(  f_{x_{1}}(u_{1_{K}}) / (u_{1_{K}}) \right)  / x_{1}  + \dots + \sum_{k=1}^{M_{N}} \left(  f_{x_{N}}(u_{N_{K}}) / (u_{N_{K}}) \right)  / x_{N} $ \\ 
\end{tabular} 
\end{center}

Note: + denotes union in the above equation.

Regarding secondary grade it is important to note that in

$$\tilde{A} = \{(x, u), \mu_{ \tilde{A} }(x, u)|\forall x\in X, \forall u \in J_{x} \subseteq [0,1]\}$$

$\mu(x', u')(x \in X, u' \in J_{x'})$ is a secondary grade.

#### Principal membership function

In cases where for each input only one primary membership has a secondary membership equal to 1, the set of all primary memberships which have a secondary membership equal to 1 is called the **principal membership function**

#### Footprint of Uncertainty

The 2D support of $\mu$ is called the **footprint of uncertainty (FOU)**

$$FOU(\tilde{A})= \left\lbrace  (x,u) \in X \times [0,1]  | \mu_{\tilde{A}}(x,u) > 0 \right\rbrace $$

FOU represents the uncertainty in the primary memberships of $\tilde{A}$. It is the union of all primary memberships.

$$FOU(\tilde{A}) = \bigcup\limits_{x\in X} J_{x}$$


\begin{figure}[ht]
	\centering
	\includegraphics[width=0.7\linewidth]{chapters/02_type2_fuzzy/img/img_3}
	\caption[Footprint of Uncertainty]{Footprint of Uncertainty}
	\label{fig:02:img3}
\end{figure}

The shaded FOU implies a distribution that sits in top of the type-2 fuzzy set in the third dimension. The distribution depends on the choice of the secondary grades. When the secondary grades are equal to 1, the sets are called **interval type-2 fuzzy sets**.

The footprint of uncertainty can be also described in terms of the upper and lower membership functions:

The **lower membership function** 

$$LMF(\tilde{A}) = \underline{\mu_{\tilde{A}}} = \inf\left\lbrace u | u\in[0,1], \mu_{ \tilde{A} }(x,u) >0 \right\rbrace $$

The **upper membership function**

$$LMF(\tilde{A}) = \overline{\mu_{\tilde{A}}} = \sup\left\lbrace u | u\in[0,1], \mu_{ \tilde{A} }(x,u) >0 \right\rbrace $$

#### Embedded Type-2 Fuzzy Sets

For discrete universes of discourse $X$ and $U$, an **embedded type-2 set**  $\tilde{A_e}$ has $N$ elements, where $\tilde{A_e}$ has exactly one element from $J_{x_{1}}, J_{x_{2}}, \dots , J_{x_{N}}$; namely $u_{1}, u_{2}, \dots , u_{N}$ each with associated grade namely $f_{x_{1}}(u_1), f_{x_{2}}(u_2), \dots , f_{x_{N}}(u_N)$, such that:

$$\tilde{A_e} = \displaystyle \sum_{i=1}^{N} \left[ f_{x_{i}} (u_{i}) \right] / x_{i}$$

where $u_{i} \in J_{x_{i}} \subseteq [0,1]$

Set $\tilde{A_e}$ is embedded in $\tilde{A}$ and there are a total of:

$$Num(\tilde{A_e}) = n = \displaystyle \prod_{i=1}^{N} M_i$$

#### Wavy Slice Representation

The wavy slice representation of a type-2 fuzzy set is given when we take the union of all embedded type-2 fuzzy sets. Given that $\tilde{A_{e}^{j}} = \{ (u_{i}^{j}, f_{x_{i}}(u_{i}^{j})), i=1\dots N \}$ and $u_{i}^{j} \in \{ u_{i_{k}}, k=1\dots M\}$, we can say

$$\tilde{A}=\sum_{j=1}^{n} \tilde{A_{e}^{j}}$$

where n is given in equation \ref{chapter1_number_of_embedded}

#### Embedded Type-1 Fuzzy Sets

For discrete universes of discourse $X$ and $U$, an **embedded type-1 set** $A_e$ has $N$ elements, where $\tilde{A_e}$ has exactly one element from $J_{x_{1}}, J_{x_{2}}, \dots , J_{x_{N}}$; namely $u_{1}, u_{2}, \dots , u_{N}$, such that:

$$A_e = \displaystyle \sum_{i=1}^{N}  u_{i}  / x_{i}$$

where $u_{i} \in J_{x_{i}} \subseteq [0,1]$

Set $A_e$ is the union of all the primary membership of $\tilde{A_e}$ and there is a a total of:

$$Num(A_e) = \displaystyle \prod_{i=1}^{N} M_i$$

**An embedded T1FS** can be also defined as a function whose range is a subset of $[0,1]$ determined by $\mu_{\tilde{A}}(x,u)$

$$A_{e}=\{x, u(x) | x\in X, u\in J_{x}\}$$

#### Type-1 Fuzzy Sets

A type-1 fuzzy set can be represented as a type-2 fuzzy set. Its type-2 representation is: $ \left( 1/ \mu(x) \right) / x $ or $ 1/\mu_{F}(x), \forall x\in X$.\\
$1/ \mu_{F}(x)$ means that the secondary membership function has only one value in its domain, i.e. the primary membership $\mu_F(x)$ at  which the secondary grade is equal to 1.
