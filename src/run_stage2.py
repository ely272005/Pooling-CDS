"""Smoke test for Stages 1-2: do the three models behave as theory predicts?

Run on a deliberately sparse panel, since that is where they separate. The
checks below are the ones that would catch a broken implementation:

  1. no-pooling is the worst on the sparsest names (variance explodes)
  2. complete pooling is flat in n (it ignores n entirely)
  3. partial pooling beats both overall
  4. partial pooling -> complete pooling as n -> 0 (the sparsity floor)
  5. the EM and MCMC versions agree (two routes to the same estimator)
"""

from __future__ import annotations

import numpy as np

from models import (
    fit_complete_pooling,
    fit_no_pooling,
    fit_partial_pooling,
    fit_partial_pooling_em,
    score,
    summarise,
)
from simulate import SimConfig, simulate

# Sparse panel: mean 3% quoting over 250 days -> single-digit obs for many
# names. This is the regime the project is about.
CFG = SimConfig(
    n_names=250,
    n_days=250,
    obs_prob_mean=0.03,
    obs_prob_spread=0.35,
    idio_sd=0.25,
    min_obs=1,
    seed=0,
)


def main() -> None:
    data = simulate(CFG)
    print("=" * 70)
    print("PANEL")
    print("=" * 70)
    print(data.summary())

    counts = data.truth["n_obs"]
    print(
        f"names with <5 obs: {(counts < 5).sum()} / {len(counts)} "
        f"({(counts < 5).mean():.0%})"
    )

    print()
    print("=" * 70)
    print("MODELS")
    print("=" * 70)

    ests = [
        fit_complete_pooling(data.obs, data.truth),
        fit_no_pooling(data.obs, data.truth),
        fit_partial_pooling_em(data.obs, data.truth),
    ]
    print("fitted: complete, nopool, partial_em")

    print("sampling partial pooling (MCMC)...")
    ests.append(
        fit_partial_pooling(data.obs, data.truth, draws=500, tune=500, chains=2)
    )
    print("fitted: partial (MCMC)")

    scored = score(ests, data.truth)
    models = ["complete", "nopool", "partial_em", "partial"]

    print()
    print("=" * 70)
    print("RMSE BY SPARSITY BUCKET (log-spread units)")
    print("=" * 70)
    summary = summarise(scored, models)
    pivot = summary.pivot(index="bucket", columns="model", values="rmse")
    order = ["all", "0-5", "5-10", "10-25", "25-100", "100-+"]
    pivot = pivot.reindex([b for b in order if b in pivot.index])
    n_by_bucket = summary.drop_duplicates("bucket").set_index("bucket")["n_names"]
    pivot.insert(0, "n_names", n_by_bucket)
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))

    print()
    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    _checks(scored, pivot)


def _checks(scored, pivot) -> None:
    def ok(cond: bool) -> str:
        return "PASS" if cond else "**FAIL**"

    rmse = {m: float(np.sqrt(np.nanmean(scored[f"err_{m}"] ** 2))) for m in
            ["complete", "nopool", "partial_em", "partial"]}

    # 3. partial beats both baselines overall
    print(
        f"{ok(rmse['partial'] < rmse['complete'])} "
        f"partial ({rmse['partial']:.4f}) < complete ({rmse['complete']:.4f})"
    )
    print(
        f"{ok(rmse['partial'] < rmse['nopool'])} "
        f"partial ({rmse['partial']:.4f}) < nopool ({rmse['nopool']:.4f})"
    )

    # 5. the two partial-pooling implementations agree
    gap = abs(rmse["partial"] - rmse["partial_em"])
    print(f"{ok(gap < 0.02)} EM vs MCMC agree (gap {gap:.4f})")

    # 1. no-pooling worst where data is thinnest
    if "0-5" in pivot.index:
        row = pivot.loc["0-5"]
        print(
            f"{ok(row['nopool'] > row['partial'])} "
            f"on <5 obs: nopool ({row['nopool']:.3f}) worse than "
            f"partial ({row['partial']:.3f})"
        )
        # 4. the sparsity floor: partial collapses toward complete
        ratio = row["partial"] / row["complete"]
        print(
            f"{'INFO'} on <5 obs: partial/complete = {ratio:.3f} "
            f"({'collapsed to group mean' if ratio > 0.95 else 'still adding info'})"
        )

    # 2. Complete pooling uses no name-specific data, so its error should be
    #    the name's true distance from its group mean -- and nothing else.
    #    (Checking RMSE is flat across buckets does NOT test this: with only
    #    ~25 names per bucket, the draw of idiosyncrasies varies enough to
    #    move RMSE around for reasons that have nothing to do with n.)
    #    err = group_mean - true_level = -(true_level - group_mean), so the
    #    correlation is NEGATIVE by construction.
    dist = scored["true_level"] - scored["group_level_truth"]
    corr = float(np.corrcoef(scored["err_complete"], dist)[0, 1])
    print(f"{ok(corr < -0.8)} complete pooling error == -(distance from group "
          f"mean) (corr {corr:.4f})")
    # It is -0.85 rather than -1.0 because the ESTIMATED group mean is itself
    # noisy -- groups are built from sparse names. That is a second channel
    # by which pooling degrades, distinct from idiosyncrasy: the anchor a
    # name is shrunk toward is uncertain. Worth separating in the sweeps.
    anchor_err = float(
        np.sqrt(np.nanmean((scored["est_complete"] - scored["group_level_truth"]) ** 2))
    )
    print(f"INFO group-mean estimation error: {anchor_err:.4f} "
          f"(vs idio_sd {CFG.idio_sd:.2f}) -- the anchor is itself uncertain")

    # And the sparsity floor, stated directly: on the sparsest names,
    # standalone estimation should be no better than ignoring the name.
    sparse = scored[scored["n_obs"] < 5]
    if len(sparse) > 5:
        r_np = float(np.sqrt(np.nanmean(sparse["err_nopool"] ** 2)))
        r_cp = float(np.sqrt(np.nanmean(sparse["err_complete"] ** 2)))
        print(
            f"{ok(r_np >= r_cp * 0.9)} on <5 obs: standalone has broken down "
            f"(nopool {r_np:.3f} vs complete {r_cp:.3f})"
        )


if __name__ == "__main__":
    main()
