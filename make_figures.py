"""Regenerate the README results figure from the deterministic ECL engine.

Offline: rebuilds the same seven-loan sample book used in Phase 3/4 and runs
the engine (S&P PD anchors + EBA-calibrated scenarios are hard-coded, so no
network), then plots ECL by IFRS 9 stage and the probability-weighted ECL
across baseline/adverse/upside.

    pip install matplotlib
    python make_figures.py        # writes docs/ifrs9_results.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecl_engine_phase3 import StagedLoan, compute_portfolio_ecl
from ecl_engine_phase4 import compute_weighted_ecl

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
os.makedirs(DOCS, exist_ok=True)


def sample_book():
    return [
        StagedLoan(name="AlphaIndustries SA", principal=2_000_000,
                   origination_rating="BBB", current_rating="BBB",
                   lgd=0.45, eir=0.035, maturity=5, amort_type="bullet", region="europe"),
        StagedLoan(name="BetaLogistics GmbH", principal=3_000_000,
                   origination_rating="A", current_rating="A-",
                   lgd=0.45, eir=0.030, maturity=7, amort_type="amortising", region="europe"),
        StagedLoan(name="GammaRetail SAS", principal=5_000_000,
                   origination_rating="BBB", current_rating="BB-",
                   lgd=0.45, eir=0.045, maturity=5, amort_type="bullet", region="europe"),
        StagedLoan(name="DeltaConstruction SpA", principal=1_500_000,
                   origination_rating="BB", current_rating="BB",
                   lgd=0.45, eir=0.050, maturity=4, days_past_due=35,
                   amort_type="bullet", region="europe"),
        StagedLoan(name="EpsilonEnergy BV", principal=4_000_000,
                   origination_rating="BBB+", current_rating="BBB-",
                   lgd=0.45, eir=0.040, maturity=6, qualitative_flags=["watchlist"],
                   amort_type="bullet", region="europe"),
        StagedLoan(name="ZetaServices SARL", principal=800_000,
                   origination_rating="BB", current_rating="CCC/C",
                   lgd=0.45, lgd_downturn=0.65, eir=0.060, maturity=3, days_past_due=95,
                   amort_type="bullet", region="europe"),
        StagedLoan(name="EtaHoldings AG", principal=2_500_000,
                   origination_rating="BB+", current_rating="B-",
                   lgd=0.45, lgd_downturn=0.65, eir=0.055, maturity=5,
                   qualitative_flags=["distressed_restructuring"],
                   amort_type="bullet", region="europe"),
    ]


def main():
    loans = sample_book()
    totals = compute_portfolio_ecl(loans)["totals"]
    weighted = compute_weighted_ecl(loans)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A — ECL by stage (with coverage % annotated)
    stages = ["Stage 1", "Stage 2", "Stage 3"]
    ecl = [totals["stage1_ecl"], totals["stage2_ecl"], totals["stage3_ecl"]]
    cov = [totals["stage1_coverage"], totals["stage2_coverage"], totals["stage3_coverage"]]
    colours = ["#4a7a96", "#d8a13a", "#b5462f"]
    bars = ax1.bar(stages, [e / 1e3 for e in ecl], color=colours)
    for b, c in zip(bars, cov):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{c:.1f}% cov", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("ECL (EUR thousands)")
    ax1.set_title("ECL by IFRS 9 stage")
    ax1.grid(alpha=0.25, axis="y")

    # Panel B — probability-weighted ECL across scenarios
    names = [s["scenario"] for s in weighted["scenarios"]]
    probs = [s["probability"] for s in weighted["scenarios"]]
    scen_ecl = [s["totals"]["total_ecl"] / 1e3 for s in weighted["scenarios"]]
    sc = ax2.bar(names, scen_ecl, color=["#4a7a96", "#b5462f", "#5a9367"])
    for b, p in zip(sc, probs):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"p={p:.0%}", ha="center", va="bottom", fontsize=9)
    w = weighted["weighted_ecl"] / 1e3
    ax2.axhline(w, color="#1f1f1f", ls="--", lw=1.5,
                label=f"Probability-weighted ECL = EUR {w:,.0f}k")
    ax2.set_ylabel("Portfolio ECL (EUR thousands)")
    ax2.set_title(f"Forward-looking ECL  (+{weighted['overlay_pct']:.0f}% overlay vs baseline)")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = os.path.join(DOCS, "ifrs9_results.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
