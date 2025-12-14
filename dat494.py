import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns

import pymc as pm
import arviz as az

# ---------------- Settings ----------------
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "SPY", "TLT"]
start = "2019-01-01"
end = "2024-10-31"

output_dir = "bayes_portfolio_outputs"
os.makedirs(output_dir, exist_ok=True)

scale_factor = 1.0

# Sampling settings
draws = 1000
tune = 2000
chains = 4
target_accept = 0.95
random_seed = 12345

# ---------------- Download prices ----------------
prices_list = []
print("Downloading prices from Stooq...")
for t in tickers:
    try:
        df = pdr.DataReader(t, "stooq", start, end)
        df = df.sort_index()
        prices_list.append(df["Close"].rename(t))
        print("OK", t)
    except Exception as e:
        print("FAILED", t, e)

prices = pd.concat(prices_list, axis=1)
print("Prices shape:", prices.shape)
prices.to_csv(os.path.join(output_dir, "prices.csv"))

# ---------------- Returns ----------------
returns = np.log(prices / prices.shift(1)).dropna()
print("Returns shape:", returns.shape)
R = returns.values  # T x K
asset_names = returns.columns.tolist()
T, K = R.shape
returns.to_csv(os.path.join(output_dir, "returns.csv"))

# optional scaling
R_scaled = R * scale_factor

# ---------------- PyMC model ----------------
print("\nBuilding PyMC model (non-centered mu) ...")
with pm.Model() as model:
    # Non-centered hierarchical mean
    mu0 = pm.Normal("mu0", mu=0.0, sigma=1.0)
    sigma0 = pm.HalfNormal("sigma0", sigma=0.05)
    mu_z = pm.Normal("mu_z", mu=0.0, sigma=1.0, shape=K)
    mu = pm.Deterministic("mu", mu0 + mu_z * sigma0)

    # LKJ Cholesky covariance
    eta = 2.0
    sd_dist = pm.HalfNormal.dist(sigma=0.05)
    L, R_corr, stds = pm.LKJCholeskyCov(
        "lkj",
        n=K,
        eta=eta,
        sd_dist=sd_dist,
        compute_corr=True
    )

    # Full covariance
    Sigma = pm.Deterministic("Sigma", L @ L.T)

    # Likelihood
    obs = pm.MvNormal("obs", mu=mu, chol=L, observed=R_scaled)

    try:
        pm.Deterministic("log_likelihood", obs.logp_elemwise)
    except Exception:
        # fallback: compute logp via aano or warn; ArviZ may still work in many cases
        print("Warning: obs.logp_elemwise not available for this PyMC version. WAIC/LOO may fail.")

    # Sampling
    idata = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        return_inferencedata=True
    )

# Save posterior
az.to_netcdf(idata, os.path.join(output_dir, "posterior_trace.nc"))
print("Saved posterior_trace.nc")

# Quick ArviZ summary
print("\nArviZ summary (mu, mu0, sigma0):")
print(az.summary(idata, var_names=["mu", "mu0", "sigma0"]).round(3))

# ---------------- Extract posterior draws ----------------
posterior = idata.posterior
chains_dim = posterior.dims.get("chain", 1)
draws_dim = posterior.dims.get("draw", 1)
n_draws = int(chains_dim * draws_dim)

mu_draws = np.empty((n_draws, K))
Sigma_draws = np.empty((n_draws, K, K))
idx = 0
for c in range(chains_dim):
    for d in range(draws_dim):
        mu_draws[idx] = posterior["mu"].values[c, d]
        Sigma_draws[idx] = posterior["Sigma"].values[c, d]
        idx += 1

print(f"Extracted {n_draws} posterior draws")

# If you scaled data, rescale draws back
if scale_factor != 1.0:
    mu_draws = mu_draws / scale_factor
    Sigma_draws = Sigma_draws / (scale_factor**2)

# ---------------- Posterior-optimal weights ----------------
weights = np.full((n_draws, K), np.nan)
for i in range(n_draws):
    S = Sigma_draws[i] + 1e-10 * np.eye(K)
    m = mu_draws[i]
    try:
        w_raw = np.linalg.solve(S, m)
        s = np.sum(w_raw)
        if np.isclose(s, 0):
            weights[i] = w_raw
        else:
            weights[i] = w_raw / s
    except Exception:
        weights[i] = np.nan

weight_mean = np.nanmean(weights, axis=0)
weight_lo = np.nanpercentile(weights, 2.5, axis=0)
weight_hi = np.nanpercentile(weights, 97.5, axis=0)

df_weights = pd.DataFrame({
    "mean": weight_mean,
    "lo": weight_lo,
    "hi": weight_hi
}, index=asset_names)
df_weights.to_csv(os.path.join(output_dir, "posterior_weights_summary.csv"))
print("Saved posterior_weights_summary.csv")

# ---------------- Classical MVO ----------------
sample_mu = R.mean(axis=0)
sample_S = np.cov(R.T, bias=False)
w_class_raw = np.linalg.solve(sample_S + 1e-10 * np.eye(K), sample_mu)
s_class = np.sum(w_class_raw)
if np.isclose(s_class, 0):
    w_class = w_class_raw
else:
    w_class = w_class_raw / s_class

df_class = pd.DataFrame({"classical": w_class}, index=asset_names)
df_class.to_csv(os.path.join(output_dir, "classical_weights.csv"))
print("Saved classical_weights.csv")

# ---------------- Main plots ----------------
# Posterior mu intervals
mu_mean = mu_draws.mean(axis=0)
mu_lo = np.percentile(mu_draws, 2.5, axis=0)
mu_hi = np.percentile(mu_draws, 97.5, axis=0)

plt.figure(figsize=(10, 4))
plt.errorbar(asset_names, mu_mean, yerr=[mu_mean - mu_lo, mu_hi - mu_mean], fmt="o")
plt.title("Posterior Mean Daily Returns")
plt.ylabel("Daily log return")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "posterior_mu_ci.png"))
plt.close()

# Correlation heatmap
Sigma_mean = Sigma_draws.mean(axis=0)
Dinv = np.diag(1.0 / np.sqrt(np.diag(Sigma_mean)))
corr = Dinv @ Sigma_mean @ Dinv

plt.figure(figsize=(7, 6))
sns.heatmap(corr, xticklabels=asset_names, yticklabels=asset_names,
            annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Posterior Mean Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "posterior_corr.png"))
plt.close()

# Violin of weights
w_df = pd.DataFrame(weights, columns=asset_names)
plt.figure(figsize=(10, 5))
sns.violinplot(data=w_df)
plt.title("Posterior Distribution of Portfolio Weights")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "weights_violin.png"))
plt.close()

# Bayes vs Classical bar plot
comp = pd.DataFrame({"Bayes": weight_mean, "Classical": w_class}, index=asset_names)
plt.figure(figsize=(9, 4))
comp.plot.bar(rot=45)
plt.title("Bayesian vs Classical Portfolio Weights")
plt.ylabel("Weight")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bayes_vs_classical.png"))
plt.close()

print("Saved main plots")

# ---------------- Diagnostics, WAIC/LOO, PPC ----------------
diag_lines = []

# R-hat, ESS
try:
    diag_lines.append("ArviZ summary for selected vars:\n")
    diag_lines.append(str(az.summary(idata, var_names=["mu", "mu0", "sigma0"], round_to=3)))
except Exception as e:
    diag_lines.append("Could not produce ArviZ summary: " + str(e))

# Divergences
div_count = None
try:
    if "diverging" in idata.sample_stats:
        div_count = int(idata.sample_stats["diverging"].values.sum())
    else:
        div_count = None
except Exception:
    div_count = None
diag_lines.append(f"\nDivergences: {div_count}\n")
print("Divergences:", div_count)

# WAIC and LOO
waic_res = None
loo_res = None
try:
    waic_res = az.waic(idata)
    diag_lines.append("\nWAIC:\n" + str(waic_res) + "\n")
except Exception as e:
    diag_lines.append("\nWAIC failed: " + str(e) + "\n")

try:
    loo_res = az.loo(idata)
    diag_lines.append("\nLOO:\n" + str(loo_res) + "\n")
except Exception as e:
    diag_lines.append("\nLOO failed: " + str(e) + "\n")

# Write diagnostics
with open(os.path.join(output_dir, "diagnostics.txt"), "w") as f:
    f.write("\n".join(map(str, diag_lines)))
print("Saved diagnostics.txt")

# Posterior predictive draws (robust extraction)
print("Sampling posterior predictive draws...")
with model:
    try:
        ppc_raw = pm.sample_posterior_predictive(idata, var_names=None, random_seed=random_seed)
    except Exception:
        try:
            ppc_raw = pm.sample_posterior_predictive(posterior, var_names=None, random_seed=random_seed)
        except Exception as e:
            raise RuntimeError("Posterior predictive sampling failed: " + str(e))

# normalize to array ppc_obs with shape (n_pp_samples, T, K)
ppc_obs = None
if hasattr(ppc_raw, "posterior_predictive"):
    ppc_vars = list(ppc_raw.posterior_predictive.data_vars)
    print("posterior_predictive variables:", ppc_vars)
    if "obs" in ppc_vars:
        ppc_obs = ppc_raw.posterior_predictive["obs"].values
    else:
        v = ppc_vars[0]
        print(f"Warning: 'obs' not found, using '{v}' for PPC")
        ppc_obs = ppc_raw.posterior_predictive[v].values
elif isinstance(ppc_raw, dict):
    print("ppc dict keys:", list(ppc_raw.keys()))
    if "obs" in ppc_raw:
        ppc_obs = np.asarray(ppc_raw["obs"])
    else:
        k0 = list(ppc_raw.keys())[0]
        print(f"Warning: 'obs' not found in ppc dict, using '{k0}'")
        ppc_obs = np.asarray(ppc_raw[k0])

if ppc_obs is None:
    raise RuntimeError("Could not extract posterior predictive draws from returned object.")

# reshape if (chain, draw, T, K)
if ppc_obs.ndim == 4:
    n_ch_pp, n_dr_pp, TT, KK = ppc_obs.shape
    ppc_obs = ppc_obs.reshape(n_ch_pp * n_dr_pp, TT, KK)
elif ppc_obs.ndim == 3:
    pass
else:
    raise RuntimeError(f"Unexpected posterior predictive array shape: {ppc_obs.shape}")

print("PPC shape:", ppc_obs.shape)

# PPC density overlays per asset
plt.figure(figsize=(12, 8))
for i, asset in enumerate(asset_names):
    plt.subplot(3, 3, i + 1)
    sns.kdeplot(returns.iloc[:, i].values, label="Observed", lw=1)
    pred_flat = ppc_obs[:, :, i].ravel()
    if pred_flat.size > 20000:
        pred_flat = np.random.choice(pred_flat, size=20000, replace=False)
    sns.kdeplot(pred_flat, label="PPC", lw=1)
    plt.title(asset)
    if i == 0:
        plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ppc_density_per_asset.png"))
plt.close()
print("Saved ppc_density_per_asset.png")

# PPC time ribbons for representative assets
rep_assets = ["SPY", "AAPL"]
for asset in rep_assets:
    if asset not in asset_names:
        continue
    idx = asset_names.index(asset)
    q05 = np.percentile(ppc_obs[:, :, idx], 5, axis=0)
    q25 = np.percentile(ppc_obs[:, :, idx], 25, axis=0)
    q50 = np.percentile(ppc_obs[:, :, idx], 50, axis=0)
    q75 = np.percentile(ppc_obs[:, :, idx], 75, axis=0)
    q95 = np.percentile(ppc_obs[:, :, idx], 95, axis=0)

    plt.figure(figsize=(12, 4))
    dates = returns.index
    plt.plot(dates, returns.iloc[:, idx], label="Observed", color="black", lw=0.8)
    plt.fill_between(dates, q05, q95, color="C0", alpha=0.15, label="5-95% PPC")
    plt.fill_between(dates, q25, q75, color="C0", alpha=0.25, label="25-75% PPC")
    plt.plot(dates, q50, color="C0", linestyle="--", label="PPC median")
    plt.title(f"PPC time series ribbon for {asset}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ppc_timeseries_{asset}.png"))
    plt.close()
    print(f"Saved ppc_timeseries_{asset}.png")

# ---------------- Save posterior summaries ----------------
mu_mean = mu_draws.mean(axis=0)
mu_lo = np.percentile(mu_draws, 2.5, axis=0)
mu_hi = np.percentile(mu_draws, 97.5, axis=0)
mu_df = pd.DataFrame({"mu_mean": mu_mean, "mu_2.5%": mu_lo, "mu_97.5%": mu_hi}, index=asset_names)
mu_df.to_csv(os.path.join(output_dir, "posterior_mu_summary.csv"))
print("Saved posterior_mu_summary.csv")

sigma_draws = np.sqrt(np.clip(np.diagonal(Sigma_draws, axis1=1, axis2=2), 0, None))
sigma_mean = sigma_draws.mean(axis=0)
sigma_lo = np.percentile(sigma_draws, 2.5, axis=0)
sigma_hi = np.percentile(sigma_draws, 97.5, axis=0)
sigma_df = pd.DataFrame({"sigma_mean": sigma_mean, "sigma_2.5%": sigma_lo, "sigma_97.5%": sigma_hi}, index=asset_names)
sigma_df.to_csv(os.path.join(output_dir, "posterior_sigma_summary.csv"))
print("Saved posterior_sigma_summary.csv")

df_weights.to_csv(os.path.join(output_dir, "posterior_weights_summary.csv"))
combined = pd.concat([mu_df, sigma_df, df_weights, df_class], axis=1)
combined.to_csv(os.path.join(output_dir, "combined_posterior_summary.csv"))
print("Saved combined_posterior_summary.csv")

# Posterior predictive mean timeline CSV
ppc_mean = ppc_obs.mean(axis=0)
ppc_mean_df = pd.DataFrame(ppc_mean, index=returns.index, columns=asset_names)
ppc_mean_df.to_csv(os.path.join(output_dir, "ppc_mean_returns.csv"))
print("Saved ppc_mean_returns.csv")

# Observed vs PPC mean bar
obs_mean = returns.mean(axis=0).values
ppc_mean_overall = ppc_obs.mean(axis=(0, 1))
plt.figure(figsize=(9, 4))
df_cmp = pd.DataFrame({"observed_mean": obs_mean, "ppc_mean": ppc_mean_overall}, index=asset_names)
df_cmp.plot.bar(rot=45)
plt.title("Observed mean vs PPC mean (across time and draws)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "obs_vs_ppc_mean.png"))
plt.close()
print("Saved obs_vs_ppc_mean.png")

print("\nAll outputs saved in:", output_dir)
print("Key files: posterior_trace.nc, diagnostics.txt, posterior_mu_summary.csv, posterior_sigma_summary.csv, posterior_weights_summary.csv, classical_weights.csv, ppc_density_per_asset.png, ppc_timeseries_*.png")
