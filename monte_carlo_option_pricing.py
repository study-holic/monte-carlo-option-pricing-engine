import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Black–Scholes analytical solution
def bs_call_price(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


# Exact simulation of S_T in one step (GBM has a closed-form terminal distribution)
def terminal_prices(S0, r, sigma, T, Z):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_terminal(S0, r, sigma, T, M, seed=None):
    Z = _rng(seed).standard_normal(M)
    return terminal_prices(S0, r, sigma, T, Z)


# Plain Monte Carlo
def mc_european_call(S0, K, r, sigma, T, M, seed=None):
    ST = simulate_terminal(S0, r, sigma, T, M, seed)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    stderr = np.exp(-r*T) * np.std(payoff, ddof=1) / np.sqrt(M)
    return price, stderr


# Variance reduction: Antithetic variates
def mc_antithetic(S0, K, r, sigma, T, M, seed=None):
    half = M // 2
    Z = _rng(seed).standard_normal(half)

    ST1 = terminal_prices(S0, r, sigma, T, Z)
    ST2 = terminal_prices(S0, r, sigma, T, -Z)

    payoff = 0.5 * (np.maximum(ST1 - K, 0) + np.maximum(ST2 - K, 0))
    price = np.exp(-r*T) * np.mean(payoff)
    stderr = np.exp(-r*T) * np.std(payoff, ddof=1) / np.sqrt(half)
    return price, stderr


# Variance reduction: Control variates
def mc_control_variate(S0, K, r, sigma, T, M, seed=None):
    ST = simulate_terminal(S0, r, sigma, T, M, seed)

    # Payoff
    payoff = np.maximum(ST - K, 0)

    # Control variate: stock price at maturity (its expectation is S0*exp(rT))
    control = ST
    control_mean = S0 * np.exp(r*T)

    cov = np.cov(payoff, control)
    c_opt = -cov[0, 1] / cov[1, 1]

    adjusted_payoff = payoff + c_opt * (control - control_mean)

    price = np.exp(-r*T) * np.mean(adjusted_payoff)
    stderr = np.exp(-r*T) * np.std(adjusted_payoff, ddof=1) / np.sqrt(M)
    return price, stderr


# Greeks (Pathwise method)
def mc_delta(S0, K, r, sigma, T, M, seed=None):
    ST = simulate_terminal(S0, r, sigma, T, M, seed)
    payoff_grad = (ST > K) * (ST / S0)
    delta = np.exp(-r*T) * payoff_grad.mean()
    return delta


# Gamma by central finite differences with common random numbers:
# the same Z drives all three bumped prices, so the noise cancels in the difference.
# The default bump is 1% of S0.
def mc_gamma(S0, K, r, sigma, T, M, seed=None, h=None):
    if h is None:
        h = 0.01 * S0
    Z = _rng(seed).standard_normal(M)

    def price(S):
        ST = terminal_prices(S, r, sigma, T, Z)
        return np.exp(-r*T) * np.mean(np.maximum(ST - K, 0))

    gamma = (price(S0 + h) - 2*price(S0) + price(S0 - h)) / (h**2)
    return gamma


# Convergence Plot
def convergence_plot(S0, K, r, sigma, T, max_M, seed=None):
    rng = _rng(seed)
    analytical = bs_call_price(S0, K, r, sigma, T)
    Ms = np.logspace(2, np.log10(max_M), 15).astype(int)

    estimators = {
        "MC Plain": mc_european_call,
        "MC Antithetic": mc_antithetic,
        "MC Control Variate": mc_control_variate,
    }

    plt.figure(figsize=(9, 5))
    for label, estimator in estimators.items():
        prices = [estimator(S0, K, r, sigma, T, M, rng)[0] for M in Ms]
        plt.plot(Ms, prices, marker="o", markersize=3, label=label)
    plt.axhline(analytical, color="k", linestyle="--", label="Analytical BS Price")
    plt.xscale("log")
    plt.xlabel("Number of paths (log scale)")
    plt.ylabel("Option price")
    plt.title("Monte Carlo Convergence to Black–Scholes Analytical Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #Example of use
    S0, K = 100, 100
    r, sigma, T = 0.05, 0.2, 1.0
    seed = 42

    price_plain, err_plain = mc_european_call(S0, K, r, sigma, T, 50_000, seed)
    price_anti, err_anti = mc_antithetic(S0, K, r, sigma, T, 50_000, seed)
    price_cv, err_cv = mc_control_variate(S0, K, r, sigma, T, 50_000, seed)

    print("Analytical:", bs_call_price(S0, K, r, sigma, T))
    print("Plain MC:", price_plain, "StdErr:", err_plain)
    print("Antithetic:", price_anti, "StdErr:", err_anti)
    print("Control Variate:", price_cv, "StdErr:", err_cv)

    # Greeks
    print("Delta:", mc_delta(S0, K, r, sigma, T, 50_000, seed))
    print("Gamma:", mc_gamma(S0, K, r, sigma, T, 50_000, seed))

    # Plot convergence
    convergence_plot(S0, K, r, sigma, T, 200_000, seed)
