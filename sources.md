# IFRS 9 ECL Engine — Data Sources

## PD (Probability of Default)

### Primary: S&P Global Ratings
- **Document:** "Default, Transition, and Recovery: 2024 Annual Global Corporate Default And Rating Transition Study"
- **Date:** March 27, 2025
- **Authors:** Nicole Serino, Nick W Kraemer (S&P Global Ratings Credit Research & Insights)
- **Tables used:**
  - Table 26 — Global average cumulative default rates by rating level (Y1–Y15), 1981–2024. Used for 1Y PD anchors and Phase 2 PD term structures.
  - Table 25 — European cumulative default rates by rating category (Y1–Y10), 1981–2024. Used for Europe-specific calibration.
  - Table 24 — Global average cumulative default rates by rating category (Y1–Y15). Cross-check.
  - Table 20 — 2024 one-year transition matrix by region (Global, U.S., Europe, Emerging). Used in Phase 3 for SICR transition logic.
  - Table 22 — Average one-year transition rates by region, 1981–2024. Long-run calibration for staging.
  - Table 3/9 — Annual default rates by rating category/level, 1981–2024. Historical context.
- **Access:** PDF obtained via S&P Global / Maalot (Israel affiliate, publicly hosted). Full study available via ESSEC Capital IQ subscription.
- **Key figures used:**
  - BBB 1Y PD: 0.13% (global), 0.05% (Europe)
  - BB 1Y PD: 0.44% (global), 0.36% (Europe)
  - B 1Y PD: 2.69% (global), 1.75% (Europe)
  - CCC/C 1Y PD: 26.12% (global), 26.26% (Europe)

### Supplementary: Moody's Annual Default Study
- **Document:** "Annual Default Study: Corporate Default and Recovery Rates"
- **Status:** To be sourced for cross-validation (free PDF, search Moody's website)

## LGD (Loss Given Default)

### Regulatory anchor
- **Foundation IRB LGD:** 45% senior unsecured, 75% subordinated (CRR Art. 161)
- **Source:** Regulation (EU) No 575/2013 (Capital Requirements Regulation)

### Empirical calibration (pending)
- **EBA IRB Benchmarking Reports** — annual publication showing LGD distributions across European banks
- **Banque de France Working Papers** — empirical French corporate recovery studies (search publications.banque-france.fr for "LGD")
- **Global Credit Data** — consortium summary statistics on bank recoveries

## Macroeconomic Scenarios (Phase 4)

### EBA 2025 EU-Wide Stress Test
- **Document:** "Macro-financial scenario for the 2025 EU-wide banking sector stress test"
- **Date:** January 2025
- **Prepared by:** ESRB Task Force on Stress Testing / ECB
- **URL:** https://www.eba.europa.eu/sites/default/files/2025-01/cd571cfe-b02c-4a08-9bcb-067c12238ef1/2025%20EU-wide%20stress%20test%20-%20Macro%20financial%20scenario.pdf
- **Key adverse scenario figures (EU-wide, 2025–2027):**
  - Cumulative real GDP deviation from baseline: −10.4% by end-2027
  - Cumulative real GDP contraction: −6.3%
  - Unemployment: +6.1 pp above baseline by end-2027
  - HICP inflation: 5.0% (2025), 3.5% (2026), 1.9% (2027)
  - EU equity prices: −50% in 2025, still −42% at end-2027
  - iTraxx overall 5Y: +169 bps (2025), still +88 bps (2027)
  - iTraxx sub-financials 5Y: +414 bps (2025)
  - 1Y EUR swap rate: 3.3% (2025), declining to 3.0% (2027)

### EBA 2025 Stress Test Results
- **Date:** August 2025
- **Key result:** CET1 ratio under adverse depleted by 370 bps to 12%; EU banks lost €547bn but remained resilient
- **URL:** https://www.eba.europa.eu/publications-and-media/publications/2025-eu-wide-stress-test-results

### ECB Macroeconomic Projections
- **Status:** To be sourced for baseline scenario (quarterly publication, ecb.europa.eu)

### Banque de France Macroeconomic Projections
- **Status:** To be sourced for France-specific paths

## Methodology References

- **IFRS 9:** International Financial Reporting Standard 9 — Financial Instruments (IASB, July 2014; effective Jan 2018). Free with registration at ifrs.org.
- **EBA GL/2017/06:** Guidelines on credit institutions' credit risk management practices and accounting for expected credit losses. Canonical IFRS 9 implementation guidance for European banks.
- **ACPR Notice:** ACPR notice on IFRS 9 implementation (Autorité de Contrôle Prudentiel et de Résolution). Search acpr.banque-france.fr.
- **EBA Stress Test Methodology:** "2025 EU-wide stress test — Methodological Note", July 2024. Contains IFRS 9 staging rules under stress (paragraphs 73–75).

## Portfolio Data (Phase 5)

### EBA Transparency Exercise
- **Description:** Semi-annual publication of bank-by-bank exposures by country, sector, and stage (S1/S2/S3) for 130+ European banks.
- **Use:** Benchmarking our model's portfolio ECL against reported figures from BNP, SG, Natixis, CA-CIB.
- **URL:** https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/transparency-exercise

---

*Last updated: May 2026*
