"""Recompute the paired TPR contrast from the independent Study I fold summaries.

Run python study1_paired_tpr.py replication_results.json with Python 3.10+
and NumPy 1.22+. The JSON supplies family_A, family_B and nominal_far under
primary_test. Each selected entry in families has folds with a batch ID and
at_far[operating_point]["tpr_by_attack"]["inject"].

The output is JSON on standard output. The observed contrast is the mean of
the within-batch differences. The percentile interval uses paired batches
sampled with replacement and gives every selected batch equal weight.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_DRAWS = 500
SEED = 0
INTERVAL_QUANTILES = (0.025, 0.975)


def rates_by_batch(folds: list[dict[str, Any]], operating_point: str) -> dict[int, float]:
    if not isinstance(folds, list) or not folds:
        raise ValueError("expected a nonempty list of folds")

    rates = {}
    for fold in folds:
        batch = fold["batch"]
        if isinstance(batch, str) and batch.isdecimal():
            batch = int(batch)
        if type(batch) is not int or batch in rates:
            raise ValueError("each fold must have a unique integer batch ID")
        rate = fold["at_far"][operating_point]["tpr_by_attack"]["inject"]
        if type(rate) not in (int, float) or not 0 <= rate <= 1:
            raise ValueError(f"invalid true positive rate for batch {batch}")
        rates[batch] = float(rate)
    return rates


def paired_summary(report: dict[str, Any]) -> dict[str, Any]:
    comparison = report["primary_test"]
    target = comparison["nominal_far"]
    if type(target) not in (int, float) or not 0 < target < 1:
        raise ValueError("nominal FAR must lie between zero and one")
    operating_point = f"{target:.2f}"
    if float(operating_point) != target:
        raise ValueError("nominal FAR must match the stored two-decimal operating point")

    families = report["families"]
    first = rates_by_batch(families[comparison["family_A"]]["folds"], operating_point)
    second = rates_by_batch(families[comparison["family_B"]]["folds"], operating_point)
    if first.keys() != second.keys():
        raise ValueError("the two families must report the same held-out batches")
    differences = np.array([first[b] - second[b] for b in sorted(first)], dtype=np.float64)
    count = len(differences)
    rng = np.random.default_rng(SEED)
    sampled_batches = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    bootstrap_means = differences[sampled_batches].mean(axis=1)
    interval = np.quantile(bootstrap_means, INTERVAL_QUANTILES, method="linear")
    return {
        "batches": count,
        "mean_delta_tpr": float(differences.mean()),
        "bootstrap_mean_delta_tpr": float(bootstrap_means.mean()),
        "percentile_interval": interval.tolist(),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Study I fold-summary JSON")
    args = parser.parse_args()
    try:
        with args.results.open(encoding="utf-8") as source:
            summary = paired_summary(json.load(source))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
