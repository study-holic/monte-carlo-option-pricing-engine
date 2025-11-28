import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ================================
# Black–Scholes analytical solution
# ================================
def bs_call_price(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


# =======================
# Path simulation
# =======================
def simulate_paths(S0, r, sigma, T, N, M):
    dt = T / N
    Z = np.random.normal(size=(M, N))
    increments = (r - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    return S0 * np.exp(log_paths)


# =======================
# Plain Monte Carlo
# =======================
def mc_european_call(S0, K, r, sigma, T, M, N=1_000):
    ST = simulate_paths(S0, r, sigma, T, N, M)[:, -1]
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    stderr = np.std(payoff) / np.sqrt(M)
    return price, stderr


# ======================================
# Variance reduction: Antithetic variates
# ======================================
def mc_antithetic(S0, K, r, sigma, T, M, N=1_000):
    half = M // 2
    dt = T / N

    Z = np.random.normal(size=(half, N))
    Z_neg = -Z

    def simulate(Z_samples):
        increments = (r - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * Z_samples
        log_paths = np.cumsum(increments, axis=1)
        return S0 * np.exp(log_paths)[:, -1]

    ST1 = simulate(Z)
    ST2 = simulate(Z_neg)

    payoff = 0.5 * (np.maximum(ST1 - K, 0) + np.maximum(ST2 - K, 0))
    price = np.exp(-r*T) * np.mean(payoff)
    stderr = np.std(payoff) / np.sqrt(half)
    return price, stderr


# ===================================
# Variance reduction: Control variates
# ===================================
def mc_control_variate(S0, K, r, sigma, T, M, N=1_000):
    ST = simulate_paths(S0, r, sigma, T, N, M)[:, -1]

    # Payoff
    payoff = np.maximum(ST - K, 0)

    # Control variate: stock price at maturity (its expectation is S0*exp(rT))
    control = ST
    control_mean = S0 * np.exp(r*T)

    cov = np.cov(payoff, control)[0, 1]
    var_control = np.var(control)
    c_opt = -cov / var_control

    adjusted_payoff = payoff + c_opt * (control - control_mean)

    price = np.exp(-r*T) * np.mean(adjusted_payoff)
    stderr = np.std(adjusted_payoff) / np.sqrt(M)
    return price, stderr


# =======================
# Greeks (Pathwise method)
# =======================
def mc_delta(S0, K, r, sigma, T, M, N=1_000):
    ST = simulate_paths(S0, r, sigma, T, N, M)[:, -1]
    payoff_grad = (ST > K) * (ST / S0)
    delta = np.exp(-r*T) * payoff_grad.mean()
    return delta


def mc_gamma(S0, K, r, sigma, T, M, N=1_000, h=0.01):
    price_up, _ = mc_european_call(S0+h, K, r, sigma, T, M, N)
    price_down, _ = mc_european_call(S0-h, K, r, sigma, T, M, N)
    price_0, _ = mc_european_call(S0,   K, r, sigma, T, M, N)
    gamma = (price_up - 2*price_0 + price_down) / (h**2)
    return gamma


# =======================
# Convergence Plot
# =======================
def convergence_plot(S0, K, r, sigma, T, max_M):
    analytical = bs_call_price(S0, K, r, sigma, T)
    Ms = np.logspace(2, np.log10(max_M), 15).astype(int)

    prices = []
    for M in Ms:
        price, _ = mc_antithetic(S0, K, r, sigma, T, M)
        prices.append(price)

    plt.figure(figsize=(9, 5))
    plt.plot(Ms, prices, label="MC Antithetic Price")
    plt.axhline(analytical, color="r", linestyle="--", label="Analytical BS Price")
    plt.xscale("log")
    plt.xlabel("Number of paths (log scale)")
    plt.ylabel("Option price")
    plt.title("Monte Carlo Convergence to Black–Scholes Analytical Price")
    plt.legend()
    plt.show()

# =======================
#Example of use
# =======================
S0, K = 100, 100
r, sigma, T = 0.05, 0.2, 1.0

price_plain, err_plain = mc_european_call(S0, K, r, sigma, T, 50_000)
price_anti, err_anti = mc_antithetic(S0, K, r, sigma, T, 50_000)
price_cv, err_cv = mc_control_variate(S0, K, r, sigma, T, 50_000)

print("Plain MC:", price_plain, "StdErr:", err_plain)
print("Antithetic:", price_anti, "StdErr:", err_anti)
print("Control Variate:", price_cv, "StdErr:", err_cv)

# Greeks
print("Delta:", mc_delta(S0, K, r, sigma, T, 50_000))
print("Gamma:", mc_gamma(S0, K, r, sigma, T, 30_000))

# Plot convergence
convergence_plot(S0, K, r, sigma, T, 200_000)

