"""
Stage 1: Synthetic sparse CDS panel with known ground truth.

The point of simulating is that on real CDS data you cannot observe the thing
you want to measure. You never see a name's "true" spread -- you see a Markit
composite which, for a sparse name, is partly matrix-priced from its peers.
So you cannot tell whether pooling recovered the name or merely reproduced the
interpolation. Here we set the truth, so the breakdown boundary is measurable.

Generative model, in log-spread space (spreads are positive and right-skewed;
log-spread is roughly normal and is what practitioners model):

    log s[i,t] = mu + beta[i] * F[t] + a_sector[i] + a_rating[i] + u[i] + e[i,t]

    mu          global mean log-spread
    F[t]        systematic credit factor (the market)
    beta[i]     name's loading on it
    a_sector    sector effect      -- the group structure pooling exploits
    a_rating    rating effect      --
    u[i]        name-specific deviation from its group -- IDIOSYNCRASY
    e[i,t]      observation noise

Pooling works by shrinking a name's estimate toward its (sector, rating) group.
That is free accuracy when u[i] is small, and a bias when u[i] is large. The
four knobs below each target one of the four numbers the project is after.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Groups are coarse but realistic: enough sectors to matter, IG/HY split by
# notch. Kept small so that per-group sample sizes stay interpretable.
SECTORS = ["Financials", "Energy", "Industrials", "Consumer", "TMT", "Utilities"]
RATINGS = ["AA", "A", "BBB", "BB", "B"]

# Typical 5Y senior CDS levels by rating, in bp. Anchors the simulation to
# roughly the right scale so results are readable in market terms.
RATING_BASE_BP = {"AA": 25.0, "A": 45.0, "BBB": 90.0, "BB": 250.0, "B": 500.0}


@dataclass
class SimConfig:
    """Knobs for the generator. Each maps to one of the four target numbers."""

    n_names: int = 200
    n_days: int = 500
    seed: int = 0

    # --- (1) SPARSITY: how often a name actually quotes -------------------
    # Fraction of days a name is observed. Real single-name CDS is far
    # sparser than equities; most names quote a handful of times a week at
    # best, and many go quiet for weeks.
    obs_prob_mean: float = 0.25
    obs_prob_spread: float = 0.20  # heterogeneity: some names far sparser
    min_obs: int = 3  # drop names below this; nothing can price them

    # --- (2) IDIOSYNCRASY: how far names sit from their group ------------
    # sd of u[i]. Large values mean the group label is a poor guide to the
    # name -- the fallen-angel case, where shrinkage drags you off truth.
    idio_sd: float = 0.25

    # --- (3) CONTAMINATION: matrix-priced quotes -------------------------
    # Fraction of observations that are NOT market observations but
    # interpolations from group peers. This is the centrepiece: such a quote
    # is already a function of the group, so pooling on it is circular.
    contamination: float = 0.0
    # How faithfully the interpolation tracks the group (1.0 = exactly the
    # group mean). Markit's matrix pricing uses sector/rating/region comps,
    # so contaminated quotes carry group information but no name information.
    contam_fidelity: float = 0.9

    # --- (4) REGIME: does the boundary move in a selloff? ----------------
    # A crisis window with elevated factor vol and dispersion. Pooling
    # assumes a stable factor structure; crises are where that assumption
    # is worst and where the estimates matter most.
    regime_switch: bool = False
    regime_start: float = 0.6  # as a fraction of the sample
    regime_len: float = 0.15
    regime_vol_mult: float = 3.0
    regime_disp_mult: float = 2.0

    # --- structural parameters -------------------------------------------
    factor_vol: float = 0.05  # daily vol of the systematic factor
    factor_persistence: float = 0.98  # AR(1); credit factors are persistent
    beta_mean: float = 1.0
    beta_sd: float = 0.3
    sector_sd: float = 0.20
    # Quote noise: bid-ask, stale marks, dealer disagreement. What matters is
    # its size RELATIVE to idio_sd -- that ratio decides whether a name's own
    # history or its group is the better guide, and so where the sparsity
    # floor sits. At idio_sd=0.25 the crossover on sub-5-obs names is around
    # obs_noise_sd 0.25-0.35, so the default puts the panel in the regime
    # where the models actually separate. Illiquid single-name CDS quote
    # noise of this order relative to cross-sectional dispersion is realistic.
    obs_noise_sd: float = 0.30
    sectors: list[str] = field(default_factory=lambda: list(SECTORS))
    ratings: list[str] = field(default_factory=lambda: list(RATINGS))


@dataclass
class SimData:
    """A simulated panel plus the ground truth needed to score against."""

    obs: pd.DataFrame  # long format, observed quotes only
    truth: pd.DataFrame  # per-name ground truth (one row per name)
    factor: np.ndarray  # F[t], the systematic factor path
    regime: np.ndarray  # bool per day, True inside the crisis window
    config: SimConfig

    @property
    def n_obs(self) -> int:
        return len(self.obs)

    def summary(self) -> str:
        counts = self.obs.groupby("name").size()
        contam = self.obs["is_matrix_priced"].mean()
        return (
            f"{len(self.truth)} names x {self.config.n_days} days -> "
            f"{self.n_obs:,} observations "
            f"({self.n_obs / (len(self.truth) * self.config.n_days):.1%} filled)\n"
            f"obs per name: min={counts.min()}, median={int(counts.median())}, "
            f"max={counts.max()}\n"
            f"matrix-priced: {contam:.1%} of observations"
        )


def simulate(config: SimConfig | None = None) -> SimData:
    """Generate a sparse CDS panel with known ground truth."""
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    n, T = cfg.n_names, cfg.n_days

    # --- regime path ------------------------------------------------------
    regime = np.zeros(T, dtype=bool)
    if cfg.regime_switch:
        lo = int(cfg.regime_start * T)
        hi = min(T, lo + int(cfg.regime_len * T))
        regime[lo:hi] = True

    # --- systematic factor: persistent AR(1), vol elevated in the crisis --
    factor = np.zeros(T)
    for t in range(1, T):
        vol = cfg.factor_vol * (cfg.regime_vol_mult if regime[t] else 1.0)
        factor[t] = cfg.factor_persistence * factor[t - 1] + rng.normal(0, vol)

    # --- name attributes --------------------------------------------------
    sector_idx = rng.integers(0, len(cfg.sectors), n)
    # Ratings skewed toward the middle of the distribution, as in the real
    # universe: few AA names, a heavy BBB/BB belly.
    rating_idx = rng.choice(
        len(cfg.ratings), size=n, p=_rating_weights(len(cfg.ratings))
    )

    beta = rng.normal(cfg.beta_mean, cfg.beta_sd, n)
    sector_effect = rng.normal(0, cfg.sector_sd, len(cfg.sectors))

    # Rating effect is the dominant driver of spread level, so it is set from
    # the empirical anchors rather than drawn.
    rating_effect = np.array([np.log(RATING_BASE_BP[r]) for r in cfg.ratings])

    # IDIOSYNCRASY: the name-specific deviation pooling will try to shrink.
    u = rng.normal(0, cfg.idio_sd, n)

    # Per-name observation probability. Beta-distributed so the panel has
    # both well-quoted and near-silent names, as the real universe does.
    obs_prob = _draw_obs_prob(rng, n, cfg.obs_prob_mean, cfg.obs_prob_spread)

    # --- true log-spread path per name ------------------------------------
    # true_level is the name's spread stripped of the factor: the quantity a
    # cross-sectional pricing model is trying to recover.
    true_level = rating_effect[rating_idx] + sector_effect[sector_idx] + u
    true_log_spread = true_level[:, None] + beta[:, None] * factor[None, :]

    # --- group means, used for matrix pricing -----------------------------
    # A contaminated quote is built from the name's (sector, rating) peers,
    # mirroring how matrix pricing interpolates from comparables.
    group_key = sector_idx * len(cfg.ratings) + rating_idx
    group_level = _group_means(true_level, group_key)
    group_beta = _group_means(beta, group_key)

    # --- draw observations ------------------------------------------------
    rows = _draw_observations(
        rng,
        cfg,
        n,
        T,
        obs_prob,
        true_log_spread,
        group_level,
        group_beta,
        factor,
        regime,
    )

    obs = pd.DataFrame(rows)

    # Drop names too sparse to price at all, then relabel so name ids stay
    # contiguous for the model code.
    counts = obs.groupby("name").size()
    keep = set(counts[counts >= cfg.min_obs].index)
    obs = obs[obs["name"].isin(keep)].reset_index(drop=True)

    truth = pd.DataFrame(
        {
            "name": np.arange(n),
            "sector": [cfg.sectors[i] for i in sector_idx],
            "rating": [cfg.ratings[i] for i in rating_idx],
            "sector_idx": sector_idx,
            "rating_idx": rating_idx,
            "group": group_key,
            "beta": beta,
            # --- the ground truth the models are scored against ---
            "true_level": true_level,
            "group_level": group_level,
            # How far this name sits from its group, in units of the
            # idiosyncratic sd. This is the x-axis of the idiosyncrasy sweep.
            "idio": u,
            "idio_z": u / cfg.idio_sd if cfg.idio_sd > 0 else np.zeros(n),
            "obs_prob": obs_prob,
        }
    )
    truth = truth[truth["name"].isin(keep)].reset_index(drop=True)

    # Attach realised observation counts and contamination share per name --
    # these are the two axes the breakdown boundary is mapped over.
    per_name = obs.groupby("name").agg(
        n_obs=("log_spread", "size"),
        contam_frac=("is_matrix_priced", "mean"),
    )
    truth = truth.merge(per_name, on="name", how="left")

    return SimData(obs=obs, truth=truth, factor=factor, regime=regime, config=cfg)


def _draw_observations(
    rng,
    cfg: SimConfig,
    n: int,
    T: int,
    obs_prob: np.ndarray,
    true_log_spread: np.ndarray,
    group_level: np.ndarray,
    group_beta: np.ndarray,
    factor: np.ndarray,
    regime: np.ndarray,
) -> list[dict]:
    """Sample which (name, day) cells are observed, and how each is priced."""
    rows: list[dict] = []

    # Observation mask: is this name quoted on this day at all?
    observed = rng.random((n, T)) < obs_prob[:, None]

    # Which observed cells are matrix-priced rather than real. Sparser names
    # are more likely to be interpolated, which is the case in practice --
    # and it is what makes the contamination bias correlate with sparsity.
    if cfg.contamination > 0:
        sparsity_w = 1.0 - obs_prob / max(obs_prob.max(), 1e-9)
        # Scale per-name contamination around the target, weighted toward
        # sparse names, then clip to keep it a valid probability.
        contam_p = np.clip(cfg.contamination * (0.5 + sparsity_w), 0.0, 1.0)
        is_matrix = rng.random((n, T)) < contam_p[:, None]
    else:
        is_matrix = np.zeros((n, T), dtype=bool)

    noise = rng.normal(0, cfg.obs_noise_sd, (n, T))
    # Dispersion widens in the crisis window, on top of the factor vol.
    if cfg.regime_switch:
        noise[:, regime] *= cfg.regime_disp_mult

    for i in range(n):
        days = np.flatnonzero(observed[i])
        for t in days:
            matrix_priced = bool(is_matrix[i, t])
            if matrix_priced:
                # A matrix-priced quote carries GROUP information and no
                # name information: it is the group's level and the group's
                # beta, blended toward the name only by (1 - fidelity).
                f = cfg.contam_fidelity
                level = f * group_level[i] + (1 - f) * true_log_spread[i, t]
                quoted = level + f * group_beta[i] * factor[t] + noise[i, t]
            else:
                quoted = true_log_spread[i, t] + noise[i, t]

            rows.append(
                {
                    "name": i,
                    "day": int(t),
                    "log_spread": quoted,
                    "spread_bp": float(np.exp(quoted)),
                    "factor": factor[t],
                    "is_matrix_priced": matrix_priced,
                    "regime": bool(regime[t]),
                    # kept for diagnostics: what the quote should have been
                    "true_log_spread": true_log_spread[i, t],
                }
            )
    return rows


def _rating_weights(k: int) -> np.ndarray:
    """Rating distribution skewed to the BBB/BB belly, as in the real universe."""
    w = np.array([0.08, 0.20, 0.34, 0.26, 0.12])[:k]
    return w / w.sum()


def _draw_obs_prob(rng, n: int, mean: float, spread: float) -> np.ndarray:
    """Per-name quoting frequency, Beta-distributed around `mean`.

    `spread` controls heterogeneity: at 0 every name quotes equally often, at
    high values the panel splits into well-quoted and near-silent names.
    """
    mean = float(np.clip(mean, 1e-3, 1 - 1e-3))
    if spread <= 0:
        return np.full(n, mean)
    # Beta with the requested mean; concentration falls as spread rises.
    conc = max(1.0 / (spread**2), 2.0)
    a, b = mean * conc, (1 - mean) * conc
    return np.clip(rng.beta(a, b, n), 0.005, 1.0)


def _group_means(values: np.ndarray, group_key: np.ndarray) -> np.ndarray:
    """Mean of `values` within each group, broadcast back to each name."""
    out = np.zeros_like(values, dtype=float)
    for g in np.unique(group_key):
        m = group_key == g
        out[m] = values[m].mean()
    return out


if __name__ == "__main__":
    data = simulate(SimConfig(seed=0))
    print(data.summary())
    print()
    print(data.truth.head().to_string(index=False))
