"""
The headline correlations under estimators other than the HAC sandwich.

Section 7 reports Newey-West standard errors with a moving-block
bootstrap as a second opinion. Both are kernel or resampling methods,
and it is reasonable to ask whether the conclusion depends on that
choice. This script re-estimates the same four correlations three other
ways.

**Driscoll-Kraay** is included and then set aside. It is a panel
estimator: it aggregates the score across units at each date before
applying a Bartlett kernel, which buys robustness to cross-sectional
dependence. The headline specification has one unit, so the
cross-sectional aggregate is the score itself and the estimator
coincides with Newey-West exactly -- verified numerically in
src/timeseries_stats.py. On a single time series it is not an
alternative to the HAC sandwich; it is the HAC sandwich.

**Effective sample size** (Pyper and Peterman, 1998) is a genuine
alternative. Rather than inflating the variance of a slope, it deflates
the degrees of freedom of the correlation, using the autocorrelation
functions of both series. No kernel, no bandwidth, no resampling.

**Prais-Winsten** is the parametric alternative. It models the serial
correlation as AR(1), quasi-differences both series, and tests the
transformed regression with conventional errors. It is efficient if
AR(1) is the right description and misleading if it is not -- which is
worth watching for the rate-path variable, whose overlapping two-quarter
construction implies an MA(1) rather than an AR(1).

Run from the project root:

    python analysis/21_alternative_inference.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timeseries_stats import (  # noqa: E402
    correlation_effective_n,
    correlation_hac,
    prais_winsten_correlation,
)

MERGED_FILE = (
    PROJECT_ROOT / "outputs" / "panel" / "communication_macro_quarterly.csv"
)

VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

MEASURE = "out_pi"
TARGETS = ["hicp_yoy", "d2_dfr"]
ERAS = [("1999-2011", 1999, 2011), ("2012-2025", 2012, 2025)]

LABELS = {"hicp_yoy": "HICP inflation", "d2_dfr": "Deposit rate, 2q change"}


def main() -> None:
    data = pd.read_csv(MERGED_FILE)
    data["year"] = pd.PeriodIndex(data["period"], freq="Q").year

    rows = []

    for target in TARGETS:
        for label, lo, hi in ERAS:
            sub = data[(data["year"] >= lo) & (data["year"] <= hi)]
            pair = sub[[MEASURE, target]].dropna()

            x = pair[MEASURE].to_numpy(float)
            y = pair[target].to_numpy(float)

            hac = correlation_hac(
                x, y, min_lags=1 if target == "d2_dfr" else 0
            )
            eff = correlation_effective_n(x, y)
            pw = prais_winsten_correlation(x, y)

            rows.append(
                {
                    "target": LABELS[target],
                    "era": label,
                    "n": len(pair),
                    "correlation": hac.correlation,
                    "hac_se": hac.standard_error,
                    "hac_p": hac.p_value,
                    "n_effective": eff.n_effective,
                    "effn_p": eff.p_value,
                    "pw_p": pw.p_value,
                }
            )

    table = pd.DataFrame(rows)

    print("\n=== HEADLINE CORRELATIONS UNDER THREE ESTIMATORS ===")
    print(
        "Driscoll-Kraay is omitted because it equals Newey-West when "
        "there is one unit."
    )
    print(
        table[
            ["target", "era", "n", "correlation", "n_effective",
             "hac_p", "effn_p", "pw_p"]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\n=== HOW MUCH INFORMATION IS THERE? ===")
    for _, r in table.iterrows():
        print(
            f"  {r['target']:26s} {r['era']}: {int(r['n'])} quarters "
            f"behave like {r['n_effective']:.1f} independent observations"
        )

    print("\n=== DO THE ESTIMATORS AGREE ON SIGNIFICANCE AT 5%? ===")
    for _, r in table.iterrows():
        verdicts = {
            "Newey-West": r["hac_p"] < 0.05,
            "effective-N": r["effn_p"] < 0.05,
            "Prais-Winsten": r["pw_p"] < 0.05,
        }
        agree = len(set(verdicts.values())) == 1
        print(
            f"  {r['target']:26s} {r['era']}: "
            f"{'all agree' if agree else 'DISAGREE'} -> "
            + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in verdicts.items())
        )

    table.to_csv(
        VALIDATION_DIR / "alternative_inference.csv", index=False
    )

    print(f"\nSaved to: {VALIDATION_DIR / 'alternative_inference.csv'}")


if __name__ == "__main__":
    main()
