"""
SustainaLens — Company Sustainability Calculator
Author: Zenet and AI
Description:
    A standalone tool to process company sustainability metrics,
    compute readiness scores, and flag potential greenwash risk.
"""

import numpy as np
import pandas as pd


def calculate_sustainability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with company sustainability inputs,
    compute complex metrics and return results.
    
    Required columns:
        - company, sector, revenue_usd_b, employees_k
        - scope1_mt, scope2_mt, scope3_mt
        - renewable_pct, sbti (bool), target_year, interim_2030_pct
        - capex_green_pct, cdp_score, year, last_year_total_mt
    """

    # --- Derived totals ---
    df["total_mt"] = df["scope1_mt"] + df["scope2_mt"] + df["scope3_mt"]
    df["intensity_mt_per_billion"] = df["total_mt"] / df["revenue_usd_b"]
    df["intensity_t_per_employee"] = (df["total_mt"] * 1e6) / (df["employees_k"] * 1e3)

    # --- Year-on-Year change ---
    df["yoy_change_mt"] = df["total_mt"] - df["last_year_total_mt"]
    df["yoy_change_pct"] = (df["yoy_change_mt"] / df["last_year_total_mt"]) * 100

    # --- CDP normalization ---
    cdp_map = {"A": 1.0, "A-": 0.9, "B": 0.7, "C": 0.5, "D": 0.3}
    df["cdp_norm"] = df["cdp_score"].map(cdp_map).fillna(0.5)

    # --- Ambition score ---
    ambition = []
    for _, r in df.iterrows():
        year_bonus = max(0, 2050 - r["target_year"]) / 20  # 2030 => 1.0, 2050 => 0
        interim = r["interim_2030_pct"] / 60               # 60% => 1.0 baseline
        sbti = 1.0 if r["sbti"] else 0.6
        ambition.append(np.clip(0.4*year_bonus + 0.4*interim + 0.2*sbti, 0, 1))
    df["ambition_score"] = ambition

    # --- Progress score ---
    progress = []
    for _, r in df.iterrows():
        yoy = np.interp(-r["yoy_change_pct"], [-5, 30], [0, 1])  # reward reductions
        ren = r["renewable_pct"] / 100
        capex = r["capex_green_pct"] / 35  # 35% = strong alignment
        progress.append(np.clip(0.5*yoy + 0.3*ren + 0.2*capex, 0, 1))
    df["progress_score"] = progress

    # --- Disclosure score ---
    required_cols = ["scope1_mt","scope2_mt","scope3_mt",
                     "renewable_pct","capex_green_pct","cdp_score"]
    completeness = df[required_cols].notna().mean(axis=1)
    df["disclosure_score"] = (0.7*df["cdp_norm"] + 0.3*completeness).clip(0,1)

    # --- Credibility score ---
    credibility = []
    for _, r in df.iterrows():
        align = np.interp(r["capex_green_pct"], [5, 35], [0, 1])
        penalty = max(0, r["ambition_score"] - r["progress_score"]) * 0.3
        credibility.append(
            np.clip(0.5*r["ambition_score"] + 0.3*align + 0.2*r["disclosure_score"] - penalty, 0, 1)
        )
    df["credibility_score"] = credibility

    # --- Composite readiness ---
    df["netzero_readiness"] = (
        0.25*df["ambition_score"] +
        0.30*df["progress_score"] +
        0.25*df["credibility_score"] +
        0.20*df["disclosure_score"]
    ).clip(0,1)

    # --- Greenwashing risk ---
    df["greenwash_risk"] = np.where(
        (df["ambition_score"] > 0.7) & (df["capex_green_pct"] < 12) & (df["yoy_change_pct"] > 0),
        "High",
        np.where((df["ambition_score"] > 0.6) & (df["progress_score"] < 0.45), "Medium", "Low")
    )

    return df


if __name__ == "__main__":
    # --- Example usage with one company ---
    sample_data = {
        "company": ["ExampleBank"],
        "sector": ["Banking"],
        "revenue_usd_b": [100],
        "employees_k": [200],
        "scope1_mt": [0.2],
        "scope2_mt": [0.3],
        "scope3_mt": [5.0],
        "renewable_pct": [80],
        "sbti": [True],
        "target_year": [2040],
        "interim_2030_pct": [45],
        "capex_green_pct": [15],
        "cdp_score": ["A-"],
        "year": [2024],
        "last_year_total_mt": [6.0],
    }
    df = pd.DataFrame(sample_data)
    results = calculate_sustainability_metrics(df)
    print(results.T)  # Print transposed for readability
