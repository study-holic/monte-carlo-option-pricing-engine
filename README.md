---

# Monte Carlo Option Pricing Engine

This project is a Monte Carlo engine for pricing European call options under the Black-Scholes model. It covers antithetic variates, control variates, Greeks estimation, and convergence analysis. The goal was to actually demonstrate the core quant finance techniques with clear, fully reproducible Python code, not just get a number out.

---

## 1. Overview

A European call option has the discounted expected payoff:

```
C = e^{-rT} * E[(S_T - K)^+]
```

Under the Black-Scholes assumptions, the stock price follows geometric Brownian motion (GBM):

```
dS_t = r S_t dt + σ S_t dW_t
```

with closed-form solution:

```
S_T = S_0 * exp( (r - 0.5 σ^2) T + σ sqrt(T) Z )    where Z ~ N(0,1)
```

Monte Carlo simulation approximates the expectation by generating M independent samples of S_T and averaging the payoff.

The methods here follow standard quant practice: variance reduction, checking against the analytical price, and numerical Greeks.

---

## 2. Core Methods

### 2.1 Plain Monte Carlo

The estimator:

```
C_MC = e^{-rT} * (1/M) * Σ (S_T^(i) - K)^+
```

The standard error decays as M^{-1/2}, which is slow, hence the point of the rest of this section.

### 2.2 Antithetic Variates

Using Z and -Z to generate paired paths cuts down the noise:

```
C_anti = 0.5 * ( C(Z) + C(-Z) )
```

### 2.3 Control Variates

I used S_T as a control variate here since E[S_T] is known in closed form:

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

There's a convergence plot in the project showing how the Monte Carlo estimators close in on the analytical Black-Scholes price as the number of paths grows. The x-axis is log-scaled so you can actually see what the variance reduction techniques are doing to estimator accuracy, rather than it getting lost in the scale.

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
LICENSE                           # MIT license
```

---

## 7. Purpose

This project is meant to show:

* stochastic simulation under GBM
* Monte Carlo methods for option pricing
* variance reduction techniques
* numerical Greeks
* convergence analysis
* a clear, modular implementation in Python

---

Thanks for reading :)
