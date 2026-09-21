"""
Stage 2: The three pooling models.

All three estimate the same quantity -- a name's `true_level`, i.e. its
log-spread stripped of the systematic factor. They differ only in how much
they let a name borrow from its (sector, rating) peers:

    complete pooling   every name gets its group's mean. No name-specific
                       information at all. Cannot be beaten on a name with
                       one noisy observation; cannot ever be right about a
                       name that genuinely differs from its group.

    no pooling         each name estimated from its own history alone.
                       Unbiased, but the variance explodes as n -> small.
                       This is the "can't model it standalone" baseline the
                       desk complaint is about.

    partial pooling    hierarchical Bayes. Each name's level is drawn from
                       its group's distribution, so the estimate is pulled
                       toward the group by an amount the data decides.
                       Shrinkage is the "borrowing".

The project's question is not "does partial pooling win" -- on average it
does, and that is a known result. It is WHERE it stops winning. So these are
written to be scored against ground truth per name, not in aggregate.

Shrinkage, for intuition. With n observations of within-name variance s2 and
between-name variance t2, the partial-pooled estimate is

    w * ybar_i + (1 - w) * group_mean,     w = n/s2 / (n/s2 + 1/t2)

As n falls, w -> 0 and partial pooling collapses onto complete pooling: the
sparsity floor. As a name's true level moves away from its group, that same
shrinkage becomes bias: the idiosyncrasy boundary.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# The factor is treated as observed here. On real data it would be estimated
# (a CDX/iTraxx index, or a first principal component); holding it fixed keeps
# the sweeps measuring pooling behaviour rather than factor-estimation error.
FACTOR_COL = "factor"


@dataclass
class Estimates:
    """Per-name estimates of `true_level` from one model."""

    model: str
    name: np.ndarray
    level: np.ndarray  # point estimate
    sd: np.ndarray | None = None  # posterior sd, where the model gives one

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({"name": self.name, f"est_{self.model}": self.level})
        if self.sd is not None:
            out[f"sd_{self.model}"] = self.sd
        return out


def _prepare(
    obs: pd.DataFrame, truth: pd.DataFrame, beta: str = "group"
) -> pd.DataFrame:
    """Residualise out the factor so what's left is the name's level.

    How beta is obtained matters more than it looks, because it is itself a
    pooling decision -- and using the true beta quietly removes most of what
    makes a sparse name hard:

      "true"   the simulator's beta. Diagnostic only: it hands the model
               information no real desk has, and leaves a clean
               mean-estimation problem in which even n=3 is nearly enough.
      "own"    OLS on the name's own history. The no-pooling stance, and
               unusable when n is small -- with 2 points it interpolates
               exactly, with 1 it is undefined.
      "group"  the name's (sector, rating) peers' beta, estimated jointly.
               The pooling stance, and the honest default: a name with three
               quotes genuinely cannot estimate its own loading, so it
               borrows one.

    Default is "group" so that sparse names carry the beta error they would
    carry in practice.
    """
    df = obs.merge(truth[["name", "beta", "group"]], on="name", how="left")

    if beta == "true":
        b = df["beta"].to_numpy()
    elif beta == "own":
        b = df["name"].map(_ols_beta_by(df, "name")).to_numpy()
    elif beta == "group":
        b = df["group"].map(_ols_beta_by(df, "group")).to_numpy()
    else:
        raise ValueError(f"unknown beta mode: {beta!r}")

    df["beta_used"] = b
    df["y"] = df["log_spread"] - b * df[FACTOR_COL]
    return df


def _ols_beta_by(df: pd.DataFrame, key: str) -> pd.Series:
    """Per-group OLS slope of log_spread on the factor, demeaned within name.

    Demeaning within name first removes each name's level, so the slope is
    identified from co-movement rather than from cross-sectional level
    differences. Falls back to 1.0 (the panel's mean loading) where a group
    has too little variation to identify a slope.
    """
    # Build explicitly rather than by column selection: when key == "name"
    # the two would collide into a duplicate column and groupby would see a
    # 2-D grouper.
    d = pd.DataFrame(
        {
            "k": df[key].to_numpy(),
            "nm": df["name"].to_numpy(),
            "y": df["log_spread"].to_numpy(),
            "f": df[FACTOR_COL].to_numpy(),
        }
    )
    d["y_c"] = d["y"] - d.groupby("nm")["y"].transform("mean")
    d["f_c"] = d["f"] - d.groupby("nm")["f"].transform("mean")

    # Vectorised OLS slope per key: sum(y_c * f_c) / sum(f_c^2). The sweeps
    # call this on hundreds of panels, so avoid a per-group .apply.
    d["xy"] = d["y_c"] * d["f_c"]
    d["xx"] = d["f_c"] ** 2
    agg = d.groupby("k")[["xy", "xx"]].sum()

    beta = (agg["xy"] / agg["xx"].where(agg["xx"] > 1e-12)).fillna(1.0)
    # Guard against wild slopes from near-zero factor variation.
    return beta.clip(-3.0, 5.0)


def fit_complete_pooling(
    obs: pd.DataFrame, truth: pd.DataFrame, beta: str = "group"
) -> Estimates:
    """Every name gets its group mean. The floor that pooling collapses to."""
    df = _prepare(obs, truth, beta=beta)
    group_mean = df.groupby("group")["y"].mean()
    overall = df["y"].mean()

    names = truth["name"].to_numpy()
    groups = truth.set_index("name")["group"]
    level = np.array(
        [group_mean.get(groups[nm], overall) for nm in names], dtype=float
    )
    return Estimates("complete", names, level)


def fit_no_pooling(
    obs: pd.DataFrame, truth: pd.DataFrame, beta: str = "own"
) -> Estimates:
    """Each name from its own data only. Unbiased, high variance.

    Defaults to beta="own": a true no-pooling stance estimates its own
    loading too. That is precisely what fails when n is small, which is the
    baseline the desk complaint describes.
    """
    df = _prepare(obs, truth, beta=beta)
    per_name = df.groupby("name")["y"].agg(["mean", "std", "size"])

    names = truth["name"].to_numpy()
    level = per_name["mean"].reindex(names).to_numpy()
    # Standard error of the mean; undefined for a single observation.
    sd = (per_name["std"] / np.sqrt(per_name["size"])).reindex(names).to_numpy()
    return Estimates("nopool", names, level, sd)


def fit_partial_pooling(
    obs: pd.DataFrame,
    truth: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 0,
    progressbar: bool = False,
    beta: str = "group",
) -> Estimates:
    """Hierarchical Bayes: name levels drawn from their group's distribution.

        y[i,t] ~ Normal(level[i], sigma)
        level[i] ~ Normal(group_level[g[i]], tau)
        group_level[g] ~ Normal(mu, tau_g)

    tau is the key parameter: it is estimated from the data, and it sets how
    hard each name is shrunk toward its group. Small tau (names look alike)
    means heavy borrowing; large tau means the model trusts each name's own
    history. The model discovers how much pooling the panel supports.

    Non-centred parameterisation -- with sparse names the funnel geometry is
    severe enough to bias the sampler otherwise.
    """
    import pymc as pm

    df = _prepare(obs, truth, beta=beta)

    names = truth["name"].to_numpy()
    name_pos = {nm: k for k, nm in enumerate(names)}
    groups = truth.set_index("name")["group"].reindex(names).to_numpy()
    uniq_groups = np.unique(groups)
    group_pos = {g: k for k, g in enumerate(uniq_groups)}

    obs_name_idx = df["name"].map(name_pos).to_numpy()
    name_group_idx = np.array([group_pos[g] for g in groups])
    y = df["y"].to_numpy()

    with pm.Model():
        mu = pm.Normal("mu", mu=float(np.mean(y)), sigma=2.0)

        # Between-group and within-group (between-name) dispersion.
        tau_g = pm.HalfNormal("tau_g", sigma=1.0)
        tau = pm.HalfNormal("tau", sigma=1.0)

        z_g = pm.Normal("z_g", 0.0, 1.0, shape=len(uniq_groups))
        group_level = pm.Deterministic("group_level", mu + tau_g * z_g)

        z_n = pm.Normal("z_n", 0.0, 1.0, shape=len(names))
        level = pm.Deterministic("level", group_level[name_group_idx] + tau * z_n)

        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("obs", mu=level[obs_name_idx], sigma=sigma, observed=y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=seed,
                progressbar=progressbar,
                compute_convergence_checks=False,
            )

    post = idata.posterior["level"]
    return Estimates(
        "partial",
        names,
        post.mean(dim=("chain", "draw")).to_numpy(),
        post.std(dim=("chain", "draw")).to_numpy(),
    )


def fit_partial_pooling_em(
    obs: pd.DataFrame, truth: pd.DataFrame, iters: int = 50, beta: str = "group"
) -> Estimates:
    """Empirical-Bayes partial pooling: the shrinkage formula, fit by moments.

    Same estimator as the Bayesian version in the Gaussian case, but closed
    form and ~1000x faster. The sweeps run this across hundreds of panels;
    the MCMC version is for the headline results, where the posterior sd
    matters. Agreement between the two is a useful correctness check.
    """
    df = _prepare(obs, truth, beta=beta)
    names = truth["name"].to_numpy()
    groups = truth.set_index("name")["group"].reindex(names).to_numpy()

    stats = df.groupby("name")["y"].agg(["mean", "var", "size"])
    ybar = stats["mean"].reindex(names).to_numpy()
    n_i = stats["size"].reindex(names).to_numpy().astype(float)

    # Within-name variance, pooled across names (names with n=1 contribute
    # nothing, which is exactly the case pooling has to carry).
    ss = (stats["var"] * (stats["size"] - 1)).reindex(names).to_numpy()
    dof = np.maximum((n_i - 1).sum(), 1.0)
    sigma2 = float(np.nansum(ss) / dof)

    level = np.zeros(len(names))
    sd = np.zeros(len(names))

    for g in np.unique(groups):
        m = groups == g
        yb, n = ybar[m], n_i[m]

        # Between-name variance within this group, by moment matching:
        # Var(ybar) = tau2 + sigma2/n, so subtract the sampling part.
        if m.sum() > 1:
            tau2 = float(np.nanvar(yb, ddof=1) - sigma2 * np.nanmean(1.0 / n))
        else:
            tau2 = 0.0
        tau2 = max(tau2, 1e-6)

        # Iterate: the group mean is itself a shrunk quantity.
        gm = float(np.nanmean(yb))
        for _ in range(iters):
            w = tau2 / (tau2 + sigma2 / n)  # weight on the name's own data
            gm_new = float(np.nansum(w * yb) / np.nansum(w))
            if abs(gm_new - gm) < 1e-10:
                gm = gm_new
                break
            gm = gm_new

        w = tau2 / (tau2 + sigma2 / n)
        level[m] = w * yb + (1 - w) * gm
        sd[m] = np.sqrt(w * sigma2 / n)  # posterior sd under the shrinkage

    return Estimates("partial_em", names, level, sd)


def score(estimates: list[Estimates], truth: pd.DataFrame) -> pd.DataFrame:
    """Per-name errors against known truth.

    Per-name, not aggregate, deliberately: the finding is about which names
    pooling fails on, so the scoring has to keep the name-level detail the
    sweeps slice by (n_obs, idio_z, contam_frac).
    """
    out = truth[
        ["name", "sector", "rating", "group", "true_level", "idio_z", "n_obs"]
    ].copy()
    # The group mean a name would be shrunk to, kept so scoring can check
    # complete pooling against it directly.
    out["group_level_truth"] = truth["group_level"].to_numpy()
    if "contam_frac" in truth.columns:
        out["contam_frac"] = truth["contam_frac"]

    for est in estimates:
        out = out.merge(est.to_frame(), on="name", how="left")
        out[f"err_{est.model}"] = out[f"est_{est.model}"] - out["true_level"]
        out[f"abs_err_{est.model}"] = out[f"err_{est.model}"].abs()
    return out


def summarise(scored: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """RMSE overall and by sparsity bucket -- a first look at the floor."""
    rows = []
    buckets = [(0, 5), (5, 10), (10, 25), (25, 100), (100, 10**9)]
    for model in models:
        col = f"err_{model}"
        if col not in scored.columns:
            continue
        rows.append(
            {
                "model": model,
                "bucket": "all",
                "n_names": len(scored),
                "rmse": float(np.sqrt(np.nanmean(scored[col] ** 2))),
                "bias": float(np.nanmean(scored[col])),
            }
        )
        for lo, hi in buckets:
            m = (scored["n_obs"] >= lo) & (scored["n_obs"] < hi)
            if not m.any():
                continue
            rows.append(
                {
                    "model": model,
                    "bucket": f"{lo}-{hi if hi < 10**9 else '+'}",
                    "n_names": int(m.sum()),
                    "rmse": float(np.sqrt(np.nanmean(scored.loc[m, col] ** 2))),
                    "bias": float(np.nanmean(scored.loc[m, col])),
                }
            )
    return pd.DataFrame(rows)
