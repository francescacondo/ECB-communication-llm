"""
Local projections of the policy path on the inflation outlook.

Section 7 splits the sample at 2012 and compares two correlations. That
spends the sample twice over: each era is estimated on roughly fifty
quarters, and with both series heavily autocorrelated neither half
carries much independent information.

A local projection in the sense of Jorda (2005) uses the whole sample
and estimates a profile rather than a point. For horizons h,

    R_{t+h} - R_t = a_h + b_h * out_pi_t
                    + g_h * late_t + d_h * (late_t x out_pi_t)
                    + controls + e_{t,h},

estimated separately at each h. The coefficient b_h traces how far
ahead directional inflation talk carries information about the policy
rate, and d_h is the rotation the note claims: how much more it carries
after 2012 than before.

Three things this buys over the two-sample comparison. The interaction
is estimated on all 108 quarters rather than on two halves. Controls
absorb variance that the raw correlation leaves in the residual, which
tightens the standard error for a legitimate reason rather than by
assumption. And a real effect should show a coherent shape across
horizons, which a single h cannot reveal.

One thing it does not buy is licence to choose a horizon. The profile is
reported in full. Selecting the horizon at which d_h is smallest and
reporting that p-value would be the specification search Section 7.7
warns about, and the p-values below are valid only as a set.

Errors are MA(h-1) by construction at horizon h, so the HAC bandwidth is
floored there.

Run from the project root:

    python analysis/22_local_projections.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timeseries_stats import (  # noqa: E402
    newey_west_bandwidth,
    newey_west_ols,
    standardise,
)

MERGED_FILE = (
    PROJECT_ROOT / "outputs" / "panel" / "communication_macro_quarterly.csv"
)

VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]

ERA_BREAK = 2012


def build(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["year"] = pd.PeriodIndex(data["period"], freq="Q").year
    data["late"] = (data["year"] >= ERA_BREAK).astype(float)

    data["out_pi_lag"] = data["out_pi"].shift(1)
    data["drate_lag"] = data["dfr_eop"].diff().shift(1)

    return data


def project(
    data: pd.DataFrame,
    horizon: int,
    controls: bool,
) -> dict:
    """One local projection at a given horizon."""

    frame = data.copy()

    frame["target"] = (
        frame["dfr_eop"].shift(-horizon) - frame["dfr_eop"]
    )

    columns = ["target", "out_pi", "late"]

    if controls:
        columns += ["hicp_yoy", "out_pi_lag", "drate_lag"]

    frame = frame[columns].dropna()

    y = standardise(frame["target"].to_numpy(float))
    x = standardise(frame["out_pi"].to_numpy(float))
    late = frame["late"].to_numpy(float)

    pieces = [np.ones(len(frame)), late, x, late * x]
    names = ["const", "late", "out_pi", "late_x_out_pi"]

    if controls:
        for c in ["hicp_yoy", "out_pi_lag", "drate_lag"]:
            pieces.append(standardise(frame[c].to_numpy(float)))
            names.append(c)

    X = np.column_stack(pieces)

    lags = newey_west_bandwidth(len(frame), min_lags=horizon - 1)

    fit = newey_west_ols(y, X, lags)

    i_early = names.index("out_pi")
    i_diff = names.index("late_x_out_pi")

    lo, hi = fit.confidence_interval(i_diff)

    return {
        "horizon": horizon,
        "controls": controls,
        "n": len(frame),
        "beta_early": float(fit.coefficients[i_early]),
        "beta_early_p": float(fit.p_values[i_early]),
        "beta_late": float(
            fit.coefficients[i_early] + fit.coefficients[i_diff]
        ),
        "rotation": float(fit.coefficients[i_diff]),
        "rotation_se": float(fit.standard_errors[i_diff]),
        "rotation_p": float(fit.p_values[i_diff]),
        "rotation_lo": lo,
        "rotation_hi": hi,
        "hac_lags": lags,
        "r_squared": fit.r_squared,
    }


def plot_profile(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, controls in zip(axes, [False, True]):
        sub = table[table["controls"] == controls]

        ax.axhline(0, color="0.5", linewidth=0.9)

        ax.fill_between(
            sub["horizon"],
            sub["rotation_lo"],
            sub["rotation_hi"],
            color="#4c72b0",
            alpha=0.18,
        )

        ax.plot(
            sub["horizon"],
            sub["rotation"],
            color="#4c72b0",
            marker="o",
            linewidth=1.8,
        )

        ax.set_xlabel("Horizon, quarters")
        ax.set_title(
            "With controls" if controls else "No controls", fontsize=11
        )
        ax.grid(alpha=0.25, linewidth=0.6)

    axes[0].set_ylabel("Rotation in the policy-path coefficient")

    fig.suptitle(
        "Local projections: how much more inflation talk tracks the "
        "policy path after 2012",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    data = build(pd.read_csv(MERGED_FILE))

    rows = [
        project(data, h, c) for c in (False, True) for h in HORIZONS
    ]

    table = pd.DataFrame(rows)

    for controls in (False, True):
        sub = table[table["controls"] == controls]

        print(
            f"\n=== {'WITH' if controls else 'WITHOUT'} CONTROLS ==="
        )
        print(
            sub[
                ["horizon", "n", "beta_early", "beta_late", "rotation",
                 "rotation_se", "rotation_p", "hac_lags"]
            ]
            .round(3)
            .to_string(index=False)
        )

    print("\n=== HOW MANY HORIZONS REACH 5%? ===")
    for controls in (False, True):
        sub = table[table["controls"] == controls]
        hit = sub[sub["rotation_p"] < 0.05]["horizon"].tolist()
        print(
            f"  {'with' if controls else 'without'} controls: "
            f"{len(hit)} of {len(HORIZONS)}"
            + (f" (h = {hit})" if hit else "")
        )

    print(
        "\nThe profile is reported whole. Picking the horizon with the "
        "smallest p-value\nand reporting that alone would be the "
        "specification search Section 7.7 warns of."
    )

    table.to_csv(
        VALIDATION_DIR / "local_projections.csv", index=False
    )

    plot_profile(table, FIGURE_DIR / "local_projections.png")

    print(f"\nSaved to: {VALIDATION_DIR / 'local_projections.csv'}")


if __name__ == "__main__":
    main()
