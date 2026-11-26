"""Report measured false-alarm rates from the Study II aggregate counts.

Run python study2_heldout_far.py results_v5.json with Python 3.10+.
The group list comes from summary["FROZEN_POOLED"]["g1_pass_arms"]. Each
per_arm entry supplies methods[calibration]["rates"][feature_set][detector]
with integer counts fp and n_neg for the held-out negative-labelled windows.

The CSV on standard output gives counts and percentages for the two
fold-calibrated methods. The pooled ROC row in the paper uses a nominal
operating point. The remaining rows report the measured percentages below.
"""

import argparse
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


CALIBRATIONS = ("INSAMPLE_PERFOLD", "INNERCV_PERFOLD")
MODELS = (("net_strong", "LR"), ("net_strong", "HGB"), ("net_strong", "RF"), ("combined", "LR"))
HEADER = ("calibration", "arm", "feature_set", "detector", "false_positives", "negative_windows", "far_percent")


def heldout_rows(report: dict[str, Any]) -> list[tuple[str, str, str, str, int, int, str]]:
    arms = report["summary"]["FROZEN_POOLED"]["g1_pass_arms"]
    if not isinstance(arms, list) or any(not isinstance(arm, str) for arm in arms):
        raise ValueError("expected a list of group names")
    if len(set(arms)) != len(arms):
        raise ValueError("expected distinct group names")

    rows = []
    for calibration in CALIBRATIONS:
        for arm in arms:
            rates = report["per_arm"][arm]["methods"][calibration]["rates"]
            for feature_set, detector in MODELS:
                counts = rates[feature_set][detector]
                fp, n_neg = counts["fp"], counts["n_neg"]
                if type(fp) is not int or type(n_neg) is not int or n_neg <= 0 or not 0 <= fp <= n_neg:
                    raise ValueError(f"invalid counts for {arm}, {calibration}, {feature_set}, {detector}")
                percent = Decimal(100) * fp / n_neg
                rows.append((calibration, arm, feature_set, detector, fp, n_neg, f"{percent:.3f}"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Study II aggregate threshold-sensitivity JSON")
    args = parser.parse_args()
    try:
        with args.results.open(encoding="utf-8") as source:
            rows = heldout_rows(json.load(source))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(HEADER)
    writer.writerows(rows)


if __name__ == "__main__":
    main()
