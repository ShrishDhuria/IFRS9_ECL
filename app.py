"""
IFRS 9 ECL Engine — Interactive Dashboard
==========================================
Two-tab Streamlit app:
  Tab 1: Single-loan deep dive with interactive sliders
  Tab 2: Portfolio-level analysis with pre-loaded illustrative portfolio

Charts:
  A — PD term structure curve (cumulative + marginal)
  C — Scenario comparison bars (stacked by stage)
  D — Adverse probability sensitivity line
  E — Coverage ratio heatmap by stage × scenario

Run:  streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from copy import deepcopy

from ecl_engine_phase1 import SP_1Y_PD, SP_1Y_PD_EUROPE
from ecl_engine_phase2 import (
    build_pd_term_structure, lifetime_ecl, LoanPhase2,
    CUMULATIVE_PD_TABLE, CUMULATIVE_PD_EUROPE,
)
from ecl_engine_phase3 import (
    StagedLoan, classify_stage, SICRConfig, Stage,
    compute_portfolio_ecl, RATING_SCALE, notch_distance,
)
from ecl_engine_phase4 import (
    build_scenarios, compute_weighted_ecl,
    adverse_probability_sensitivity, shift_rating,
    compute_scenario_ecl,
)

# ── Page config ──
st.set_page_config(
    page_title="IFRS 9 ECL Engine",
    page_icon="🏦",
    layout="wide",
)

# ── Dark theme styling ──
NAVY = "#0D1B2A"
DARK_BG = "#1B2838"
CARD_BG = "#243447"
ACCENT_BLUE = "#4895EF"
ACCENT_GREEN = "#06D6A0"
ACCENT_AMBER = "#FFB703"
ACCENT_RED = "#EF476F"
TEXT_LIGHT = "#E0E0E0"
TEXT_MUTED = "#8899AA"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {NAVY};
        color: {TEXT_LIGHT};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CARD_BG};
        color: {TEXT_LIGHT};
        border-radius: 6px;
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ACCENT_BLUE};
        color: white;
    }}
    .stMetric {{
        background-color: {CARD_BG};
        padding: 12px;
        border-radius: 8px;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT_LIGHT};
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
    }}
    .stSelectbox label, .stSlider label, .stRadio label {{
        color: {TEXT_LIGHT} !important;
    }}
    h1, h2, h3 {{
        color: {TEXT_LIGHT} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark style ──
def dark_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_MUTED)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT_LIGHT)
    for spine in ax.spines.values():
        spine.set_color(TEXT_MUTED)
        spine.set_linewidth(0.5)
    return fig, ax


# ── Rating options for selectors ──
RATING_OPTIONS = [r for r in RATING_SCALE if r != "D"]
PD_TABLE_RATINGS = list(CUMULATIVE_PD_EUROPE.keys())


# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align: center; padding: 20px 0 10px 0;'>
    <h1 style='color: {TEXT_LIGHT}; margin-bottom: 0;'>IFRS 9 Expected Credit Loss Engine</h1>
    <p style='color: {TEXT_MUTED}; font-size: 16px;'>
        PD calibration: S&P 2024 Global Default Study &nbsp;|&nbsp;
        Scenarios: EBA 2025 EU-wide Stress Test &nbsp;|&nbsp;
        Methodology: EBA GL/2017/06
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍  Single-Loan Explorer", "📊  Portfolio Analysis"])


# ═══════════════════════════════════════════════════════════
# TAB 1: SINGLE-LOAN EXPLORER
# ═══════════════════════════════════════════════════════════
with tab1:
    col_input, col_output = st.columns([1, 2.5])

    with col_input:
        st.markdown("### Loan Parameters")
        rating = st.selectbox("S&P Rating", PD_TABLE_RATINGS, index=3)  # BBB
        maturity = st.slider("Maturity (years)", 1, 10, 5)
        principal = st.number_input("Principal (€)", value=1_000_000,
                                     step=100_000, format="%d")
        lgd = st.slider("LGD (%)", 10, 90, 45) / 100
        eir = st.slider("EIR (%)", 0.5, 10.0, 3.5, step=0.5) / 100
        amort = st.radio("Amortisation", ["Bullet", "Amortising"])

        origination_rating = st.selectbox(
            "Origination Rating (for staging)",
            PD_TABLE_RATINGS, index=3
        )

    with col_output:
        # Build loan and compute
        loan_p2 = LoanPhase2(
            name="Interactive Loan",
            principal=principal,
            rating=rating,
            lgd=lgd,
            eir=eir,
            maturity=maturity,
            amort_type=amort.lower(),
            region="europe",
        )
        ecl_result = lifetime_ecl(loan_p2)

        staged = StagedLoan(
            name="Interactive Loan",
            principal=principal,
            origination_rating=origination_rating,
            current_rating=rating,
            lgd=lgd,
            eir=eir,
            maturity=maturity,
            amort_type=amort.lower(),
            region="europe",
        )
        staging = classify_stage(staged)
        ecl_applied = ecl_result["ecl_12m"] if staging.stage == Stage.STAGE_1 else ecl_result["ecl_lifetime"]

        # ── Metrics row ──
        st.markdown("### ECL Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stage", f"Stage {staging.stage.value}")
        m2.metric("12-Month ECL", f"€{ecl_result['ecl_12m']:,.0f}")
        m3.metric("Lifetime ECL", f"€{ecl_result['ecl_lifetime']:,.0f}")
        cliff = ecl_result['ratio_lifetime_to_12m']
        m4.metric("Cliff Effect", f"{cliff:.1f}x" if cliff else "N/A")

        # Staging trigger
        st.markdown(f"**Staging trigger:** {staging.triggers_fired[0]}")

        # ── Chart A: PD term structure ──
        st.markdown("### PD Term Structure")
        pd_curve = build_pd_term_structure(rating, maturity, "europe")

        fig, ax = dark_fig((10, 4))
        years = [p["year"] for p in pd_curve]
        cum_pds = [p["cum_pd"] * 100 for p in pd_curve]
        marg_pds = [p["marginal_pd"] * 100 for p in pd_curve]

        ax.fill_between(years, cum_pds, alpha=0.3, color=ACCENT_BLUE)
        ax.plot(years, cum_pds, color=ACCENT_BLUE, linewidth=2.5,
                marker='o', markersize=6, label='Cumulative PD')
        ax.bar([y - 0.15 for y in years], marg_pds, width=0.3,
               color=ACCENT_AMBER, alpha=0.8, label='Marginal PD')

        ax.set_xlabel("Year")
        ax.set_ylabel("PD (%)")
        ax.set_title(f"PD Term Structure — {rating} (European, S&P 1981-2024)")
        ax.legend(facecolor=CARD_BG, edgecolor=TEXT_MUTED, labelcolor=TEXT_LIGHT)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f%%'))
        ax.set_xticks(years)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════
# TAB 2: PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab2:
    # Pre-loaded portfolio
    portfolio = [
        StagedLoan("AlphaIndustries SA", 2_000_000, "BBB", "BBB",
                   0.45, 0.035, 5, "bullet", "europe"),
        StagedLoan("BetaLogistics GmbH", 3_000_000, "A", "A-",
                   0.45, 0.030, 7, "amortising", "europe"),
        StagedLoan("GammaRetail SAS", 5_000_000, "BBB", "BB-",
                   0.45, 0.045, 5, "bullet", "europe"),
        StagedLoan("DeltaConstruction SpA", 1_500_000, "BB", "BB",
                   0.45, 0.050, 4, "bullet", "europe", days_past_due=35),
        StagedLoan("EpsilonEnergy BV", 4_000_000, "BBB+", "BBB-",
                   0.45, 0.040, 6, "bullet", "europe",
                   qualitative_flags=["watchlist"]),
        StagedLoan("ZetaServices SARL", 800_000, "BB", "CCC/C",
                   0.45, 0.060, 3, "bullet", "europe",
                   days_past_due=95, lgd_downturn=0.65),
        StagedLoan("EtaHoldings AG", 2_500_000, "BB+", "B-",
                   0.45, 0.055, 5, "bullet", "europe",
                   qualitative_flags=["distressed_restructuring"],
                   lgd_downturn=0.65),
    ]

    # ── Compute weighted ECL ──
    weighted = compute_weighted_ecl(portfolio)

    # ── Header metrics ──
    st.markdown("### Probability-Weighted ECL")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weighted ECL", f"€{weighted['weighted_ecl']:,.0f}")
    m2.metric("Baseline ECL", f"€{weighted['baseline_ecl']:,.0f}")
    m3.metric("IFRS 9 Overlay", f"€{weighted['ifrs9_overlay']:,.0f}")
    m4.metric("Overlay %", f"+{weighted['overlay_pct']:.1f}%")

    # ── Chart C + Chart E side by side ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Scenario ECL by Stage")
        fig, ax = dark_fig((6, 4.5))

        scenarios = weighted["scenarios"]
        names = [s["scenario"] for s in scenarios]
        s1_vals = [s["totals"]["stage1_ecl"] / 1000 for s in scenarios]
        s2_vals = [s["totals"]["stage2_ecl"] / 1000 for s in scenarios]
        s3_vals = [s["totals"]["stage3_ecl"] / 1000 for s in scenarios]

        x = np.arange(len(names))
        width = 0.5

        bars1 = ax.bar(x, s1_vals, width, label='Stage 1',
                       color=ACCENT_GREEN, alpha=0.9)
        bars2 = ax.bar(x, s2_vals, width, bottom=s1_vals, label='Stage 2',
                       color=ACCENT_AMBER, alpha=0.9)
        s1_s2 = [a + b for a, b in zip(s1_vals, s2_vals)]
        bars3 = ax.bar(x, s3_vals, width, bottom=s1_s2, label='Stage 3',
                       color=ACCENT_RED, alpha=0.9)

        # Total labels on top
        for i, total in enumerate([a + b for a, b in zip(s1_s2, s3_vals)]):
            ax.text(i, total + 15, f'€{total:.0f}k', ha='center',
                    color=TEXT_LIGHT, fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("ECL (€ thousands)")
        ax.set_title("ECL by Stage × Scenario")
        ax.legend(facecolor=CARD_BG, edgecolor=TEXT_MUTED, labelcolor=TEXT_LIGHT,
                  loc='upper left', fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown("### Coverage Ratio (bps)")
        # Chart E: Coverage heatmap
        fig, ax = dark_fig((6, 4.5))

        stage_labels = ["Stage 1", "Stage 2", "Stage 3", "Total"]
        scn_labels = [s["scenario"] for s in scenarios]

        coverage_matrix = []
        for s in scenarios:
            t = s["totals"]
            row = [
                t["stage1_coverage"] * 100,  # convert to bps
                t["stage2_coverage"] * 100,
                t["stage3_coverage"] * 100,
                t["total_coverage"] * 100,
            ]
            coverage_matrix.append(row)

        data = np.array(coverage_matrix)
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(len(stage_labels)))
        ax.set_xticklabels(stage_labels, fontsize=10)
        ax.set_yticks(range(len(scn_labels)))
        ax.set_yticklabels(scn_labels, fontsize=10)
        ax.set_title("Coverage Ratio (bps) by Stage × Scenario")

        # Annotate cells
        for i in range(len(scn_labels)):
            for j in range(len(stage_labels)):
                val = data[i, j]
                text_color = "white" if val > 300 else TEXT_LIGHT
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        color=text_color, fontsize=11, fontweight='bold')

        fig.colorbar(im, ax=ax, label='bps', shrink=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Chart D: Sensitivity ──
    st.markdown("### Sensitivity: ECL vs Adverse Scenario Probability")

    sensitivity = adverse_probability_sensitivity(portfolio)

    fig, ax = dark_fig((10, 4))
    probs = [r["p_adverse"] * 100 for r in sensitivity]
    ecls = [r["weighted_ecl"] / 1000 for r in sensitivity]
    overlays = [r["overlay_pct"] for r in sensitivity]

    ax.plot(probs, ecls, color=ACCENT_BLUE, linewidth=2.5,
            marker='o', markersize=7, zorder=5)
    ax.fill_between(probs, ecls, alpha=0.15, color=ACCENT_BLUE)

    # Highlight chosen weight (30%)
    chosen_idx = [i for i, r in enumerate(sensitivity) if r["p_adverse"] == 0.30][0]
    ax.plot(probs[chosen_idx], ecls[chosen_idx], 'o', color=ACCENT_AMBER,
            markersize=12, zorder=6, label=f'Chosen: {probs[chosen_idx]:.0f}%')
    ax.annotate(f'€{ecls[chosen_idx]:.0f}k\n(+{overlays[chosen_idx]:.0f}% overlay)',
                xy=(probs[chosen_idx], ecls[chosen_idx]),
                xytext=(probs[chosen_idx] + 5, ecls[chosen_idx] + 30),
                color=ACCENT_AMBER, fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=ACCENT_AMBER, lw=1.5))

    ax.set_xlabel("Adverse Scenario Probability (%)")
    ax.set_ylabel("Weighted ECL (€ thousands)")
    ax.set_title("Model Risk: Weighted ECL Sensitivity to Scenario Weights")
    ax.legend(facecolor=CARD_BG, edgecolor=TEXT_MUTED, labelcolor=TEXT_LIGHT)

    # Add overlay % on secondary axis
    ax2 = ax.twinx()
    ax2.plot(probs, overlays, color=ACCENT_RED, linewidth=1.5,
             linestyle='--', alpha=0.7, label='IFRS 9 overlay %')
    ax2.set_ylabel("IFRS 9 Overlay (%)", color=ACCENT_RED)
    ax2.tick_params(axis='y', colors=ACCENT_RED)
    ax2.spines['right'].set_color(ACCENT_RED)
    ax2.legend(facecolor=CARD_BG, edgecolor=TEXT_MUTED, labelcolor=TEXT_LIGHT,
               loc='lower right')

    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Staging audit trail ──
    with st.expander("📋 Staging Audit Trail (Baseline Scenario)"):
        baseline = weighted["scenarios"][0]
        for loan in baseline["loans"]:
            stage_emoji = {1: "🟢", 2: "🟡", 3: "🔴"}
            s = loan["stage"].value
            st.markdown(
                f"{stage_emoji[s]} **{loan['loan_name']}** — "
                f"Stage {s} | "
                f"{loan['orig_rating']} → {loan['curr_rating']} | "
                f"ECL: €{loan['ecl_applied']:,.0f}"
            )
            for t in loan.get("triggers", []):
                st.caption(f"  ↳ {t}")

# ── Footer ──
st.markdown(f"""
<div style='text-align: center; padding: 20px; color: {TEXT_MUTED}; font-size: 12px;'>
    IFRS 9 ECL Engine | ESSEC MIF | Data: S&P Global Ratings 2024, EBA 2025 Stress Test |
    Methodology: EBA GL/2017/06
</div>
""", unsafe_allow_html=True)
