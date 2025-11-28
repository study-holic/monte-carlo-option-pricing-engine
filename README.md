# monte-carlo-option-pricing-engine
Monte Carlo Option Pricing Engine with variance reduction techniques (antithetic variates, control variates), Greeks estimation, and convergence analysis.

Monte Carlo Option Pricing Engine

This project implements a Monte Carlo engine for pricing European call options under the Black–Scholes model. It includes antithetic variates, control variates, Greeks estimation, and convergence analysis. The goal is to demonstrate core quantitative finance techniques using clear and fully reproducible Python code.

1. Overview

A European call option has the discounted expected payoff

𝐶
=
𝑒
−
𝑟
𝑇
 
𝐸
[
(
𝑆
𝑇
−
𝐾
)
+
]
.
C=e
−rT
E[(S
T
	​

−K)
+
].

Under the Black–Scholes assumptions, the stock price follows geometric Brownian motion (GBM):

𝑑
𝑆
𝑡
=
𝑟
𝑆
𝑡
 
𝑑
𝑡
+
𝜎
𝑆
𝑡
 
𝑑
𝑊
𝑡
,
dS
t
	​

=rS
t
	​

dt+σS
t
	​

dW
t
	​

,

with closed-form solution

𝑆
𝑇
=
𝑆
0
exp
⁡
(
(
𝑟
−
1
2
𝜎
2
)
𝑇
+
𝜎
𝑇
 
𝑍
)
,
𝑍
∼
𝑁
(
0
,
1
)
.
S
T
	​

=S
0
	​

exp((r−
2
1
	​

σ
2
)T+σ
T
	​

Z),Z∼N(0,1).

Monte Carlo simulation approximates the expectation by generating 
𝑀
M independent samples of 
𝑆
𝑇
S
T
	​

 and averaging the payoff.

The methods included here follow standard quant practice: variance reduction, comparison against analytical prices, and numerical Greeks.

2. Core Methods
2.1 Plain Monte Carlo

The basic estimator is

𝐶
^
𝑀
𝐶
=
𝑒
−
𝑟
𝑇
1
𝑀
∑
𝑖
=
1
𝑀
(
𝑆
𝑇
(
𝑖
)
−
𝐾
)
+
.
C
^
MC
	​

=e
−rT
M
1
	​

i=1
∑
M
	​

(S
T
(i)
	​

−K)
+
.

Its standard error decays as 
𝑀
−
1
/
2
M
−1/2
, which is slow.

2.2 Antithetic Variates

If 
𝑍
Z generates one path, using 
−
𝑍
−Z generates another path that counteracts sampling noise.
The estimator becomes

𝐶
^
𝑎
𝑛
𝑡
𝑖
=
1
2
(
𝐶
(
𝑍
)
+
𝐶
(
−
𝑍
)
)
.
C
^
anti
	​

=
2
1
	​

(C(Z)+C(−Z)).

This typically lowers the variance substantially.

2.3 Control Variates

We use 
𝑆
𝑇
S
T
	​

 as the control variate because its expectation is known:

𝐸
[
𝑆
𝑇
]
=
𝑆
0
𝑒
𝑟
𝑇
.
E[S
T
	​

]=S
0
	​

e
rT
.

The adjusted estimator is

𝐶
^
𝑐
𝑣
=
𝐶
^
𝑀
𝐶
+
𝑐
opt
(
𝑆
𝑇
−
𝐸
[
𝑆
𝑇
]
)
,
C
^
cv
	​

=
C
^
MC
	​

+c
opt
	​

(S
T
	​

−E[S
T
	​

]),

where

𝑐
opt
=
−
Cov
(
𝐶
,
𝑆
𝑇
)
Var
(
𝑆
𝑇
)
.
c
opt
	​

=−
Var(S
T
	​

)
Cov(C,S
T
	​

)
	​

.

This reduction is often significant, especially for European calls.

3. Greeks
Delta

Using the pathwise derivative method:

Δ
=
𝑒
−
𝑟
𝑇
 
𝐸
[
1
{
𝑆
𝑇
>
𝐾
}
 
𝑆
𝑇
𝑆
0
]
.
Δ=e
−rT
E[1
{S
T
	​

>K}
	​

S
0
	​

S
T
	​

	​

].
Gamma

Gamma is computed using a standard centered finite-difference approximation:

Γ
≈
𝐶
(
𝑆
0
+
ℎ
)
−
2
𝐶
(
𝑆
0
)
+
𝐶
(
𝑆
0
−
ℎ
)
ℎ
2
.
Γ≈
h
2
C(S
0
	​

+h)−2C(S
0
	​

)+C(S
0
	​

−h)
	​

.
4. Convergence

The project includes code to plot estimator convergence against the analytical Black–Scholes price.
A log-scaled path count illustrates how variance reduction shifts the curve upward, achieving lower error for the same computational effort.

5. Running the Code

Dependencies:

numpy
scipy
matplotlib


Run the project:

python monte_carlo_option_pricing.py


This prints:

plain Monte Carlo price

antithetic price

control variate price

Delta and Gamma

a convergence plot comparing Monte Carlo estimators to the analytical price

6. File Structure
monte_carlo_option_pricing.py     # Full implementation
README.md                         # Project documentation
images/                           # Optional: convergence plots
LICENSE                           # MIT license

7. Purpose

This project serves as a compact demonstration of:

stochastic simulation under GBM

Monte Carlo estimation for derivative pricing

variance reduction techniques used in quantitative finance

numerical estimation of Greeks

convergence analysis relative to a closed-form benchmark

clear, modular implementation in Python
