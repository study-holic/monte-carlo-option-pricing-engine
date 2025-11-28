---

# Monte Carlo Option Pricing Engine

This project implements a Monte Carlo engine for pricing European call options under the Black–Scholes model. It includes antithetic variates, control variates, Greeks estimation, and convergence analysis. The goal is to demonstrate core quantitative finance techniques using clear and fully reproducible Python code.

---

## 1. Overview

A European call option has the discounted expected payoff:

```
C = e^{-rT} * E[(S_T - K)^+]
```

Under the Black–Scholes assumptions, the stock price follows geometric Brownian motion (GBM):

```
dS_t = r S_t dt + σ S_t dW_t
```

with closed-form solution:

```
S_T = S_0 * exp( (r - 0.5 σ^2) T + σ sqrt(T) Z )    where Z ~ N(0,1)
```

Monte Carlo simulation approximates the expectation by generating M independent samples of S_T and averaging the payoff.

The methods included here follow standard quant practice: variance reduction, comparison against analytical prices, and numerical Greeks.

---

## 2. Core Methods

### 2.1 Plain Monte Carlo

The estimator:

```
C_MC = e^{-rT} * (1/M) * Σ (S_T^(i) - K)^+
```

The standard error decays as M^{-1/2}, which is slow.

### 2.2 Antithetic Variates

Using Z and −Z to generate paired paths reduces noise:

```
C_anti = 0.5 * ( C(Z) + C(-Z) )
```

### 2.3 Control Variates

Using S_T as a control variate because E[S_T] is known:

```
E[S_T] = S_0 * e^{rT}
```

Adjusted estimator:

```
C_cv = C_MC + c_opt * (S_T - E[S_T])
```

with:

```
c_opt = - Cov(C, S_T) / Var(S_T)
```

---

## 3. Greeks

### Delta (pathwise method)

```
Delta = e^{-rT} * E[ 1_{S_T > K} * (S_T / S_0) ]
```

### Gamma (finite differences)

```
Gamma = ( C(S_0 + h) - 2 C(S_0) + C(S_0 - h) ) / h^2
```

---

## 4. Convergence

The project includes a convergence plot showing how Monte Carlo estimators approach the analytical Black–Scholes price as the number of paths increases.
A log-scaled x-axis illustrates the effect of variance reduction techniques on estimator accuracy.

---

## 5. Running the Code

Dependencies:

```
numpy
scipy
matplotlib
```

Run:

```
python monte_carlo_option_pricing.py
```

This outputs:

* plain Monte Carlo price
* antithetic variates price
* control variate price
* Delta and Gamma
* a convergence plot

---

## 6. File Structure

```
monte_carlo_option_pricing.py     # Full implementation
README.md                         # Documentation
images/                           # Optional: plots for convergence
LICENSE                           # MIT license
```

---

## 7. Purpose

This project demonstrates:

* stochastic simulation under GBM
* Monte Carlo methods for option pricing
* variance reduction techniques
* numerical Greeks
* convergence analysis
* clear and modular implementation in Python

---

Thanks for reading :)
