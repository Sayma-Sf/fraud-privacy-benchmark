"""Streamlit demo: move the privacy budget and watch fraud-detection accuracy respond.

    streamlit run app.py        (on Windows: python -m streamlit run app.py)

Reads precomputed metrics from results/. No transaction data is loaded or shipped.
"""

from __future__ import annotations

import math

import altair as alt
import pandas as pd
import streamlit as st

from src.reporting.app_data import (
    INF_LABEL,
    MODEL_NAMES,
    curves_frame,
    load_results,
    plain_language,
    sweep_frame,
)

st.set_page_config(page_title="Fraud privacy benchmark", page_icon="🔒", layout="wide")

try:
    DARK = st.context.theme.type == "dark"
except AttributeError:
    DARK = False
PALETTE = {
    "series": ["#3987e5", "#d95926", "#199e70"] if DARK else ["#2a78d6", "#eb6834", "#1baf7a"],
    "muted": "#898781",
    "text": "#c3c2b7" if DARK else "#52514e",
    "surface": "#0e1117" if DARK else "#ffffff",
}


@st.cache_data
def results() -> tuple[dict, dict]:
    return load_results()


summary, curves = results()
sweep = sweep_frame(summary)
data = summary["dataset"]
lightgbm = summary["baseline"]["lightgbm"]
no_noise = sweep[sweep["label"] == INF_LABEL].iloc[0]

st.title("How much accuracy does privacy cost?")
st.markdown(
    f"A fraud model trained on **{data['rows']:,} real card transactions** "
    f"({data['frauds']} frauds, {data['fraud_rate']:.2%}). Data like this can't be shared, "
    "so there are two ways to learn from it safely: **train with differential privacy**, "
    "or **release synthetic data** instead. This page shows what each one costs in accuracy."
)

tab_dp, tab_synth, tab_why = st.tabs(
    ["Privacy budget (ε)", "Synthetic data", "Why this data can't be shared"]
)

# --- Differential privacy --------------------------------------------------------------------
with tab_dp:
    labels = sweep["label"].tolist()
    choice = st.select_slider(
        "Privacy budget ε. Slide left for more privacy.",
        options=labels,
        value="1" if "1" in labels else labels[len(labels) // 2],
    )
    row = sweep[sweep["label"] == choice].iloc[0]
    st.markdown(plain_language(row["epsilon"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "AUPRC (higher is better)",
        f"{row['auprc_mean']:.3f}",
        delta=f"{row['auprc_mean'] - no_noise['auprc_mean']:+.3f} vs no noise",
        help=f"Mean of {int(row['runs'])} training runs. A random guess scores "
        f"{data['fraud_rate']:.4f}.",
    )
    c2.metric("Accuracy kept vs no noise", f"{row['share_of_no_noise']:.0%}")
    c3.metric("Frauds caught (recall)", f"{row['recall_mean']:.0%}",
              help="At the alert threshold that maximised F1 on the validation split.")
    c4.metric("Alerts that are fraud (precision)", f"{row['precision_mean']:.0%}")

    finite = sweep[~sweep["epsilon"].map(math.isinf)]
    x = alt.X(
        "epsilon:Q",
        scale=alt.Scale(type="log"),
        axis=alt.Axis(values=[0.01, 0.1, 1, 10, 100], format="~g", grid=False),
        title="Privacy budget ε (log scale). Smaller = more private",
    )
    y_scale = alt.Scale(domain=[0, 1])
    tooltip = [
        alt.Tooltip("label:N", title="ε"),
        alt.Tooltip("auprc_mean:Q", title="Mean AUPRC", format=".3f"),
        alt.Tooltip("auprc_p10:Q", title="10th percentile", format=".3f"),
        alt.Tooltip("auprc_p90:Q", title="90th percentile", format=".3f"),
        alt.Tooltip("recall_mean:Q", title="Recall", format=".0%"),
        alt.Tooltip("precision_mean:Q", title="Precision", format=".0%"),
    ]
    blue = PALETTE["series"][0]
    base = alt.Chart(finite)
    band = base.mark_area(color=blue, opacity=0.12).encode(
        x=x, y=alt.Y("auprc_p10:Q", scale=y_scale), y2="auprc_p90:Q"
    )
    line = base.mark_line(color=blue, strokeWidth=2).encode(
        x=x, y=alt.Y("auprc_mean:Q", scale=y_scale, title="AUPRC on held-out test set")
    )
    points = base.mark_circle(color=blue, size=70, opacity=1, stroke=PALETTE["surface"],
                              strokeWidth=2).encode(x=x, y="auprc_mean:Q")
    hover = base.mark_circle(size=500, opacity=0).encode(x=x, y="auprc_mean:Q", tooltip=tooltip)
    references = pd.DataFrame({
        "value": [lightgbm["auprc"], no_noise["auprc_mean"]],
        "label": [f"LightGBM, no privacy: {lightgbm['auprc']:.2f}",
                  f"Same model, no noise (ε = ∞): {no_noise['auprc_mean']:.2f}"],
        "dy": [-8, 10],
    })
    rules = alt.Chart(references).mark_rule(color=PALETTE["muted"], strokeWidth=1).encode(
        y="value:Q", tooltip=[alt.Tooltip("label:N", title="Reference")]
    )
    rule_labels = [
        alt.Chart(references.iloc[[i]])
        .mark_text(align="left", dx=6, dy=int(r.dy), fontSize=12, color=PALETTE["text"])
        .encode(x=alt.value(0), y="value:Q", text="label:N")
        for i, r in enumerate(references.itertuples())
    ]
    layers = [band, line, rules, *rule_labels, points, hover]
    if not math.isinf(row["epsilon"]):
        layers.append(
            alt.Chart(finite[finite["label"] == choice])
            .mark_circle(size=260, color=blue, opacity=1, stroke=PALETTE["surface"], strokeWidth=3)
            .encode(x=x, y="auprc_mean:Q", tooltip=tooltip)
        )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Accuracy across privacy budgets")
        st.altair_chart(alt.layer(*layers).properties(height=360), use_container_width=True)
    with right:
        st.subheader(f"Precision vs recall at ε = {choice}")
        dp_name = f"DP model, ε = {choice}"
        curve_set = {dp_name: curves["dp"][row["epsilon_label"]]}
        if choice != INF_LABEL:
            curve_set["Same model, no noise"] = curves["dp"]["inf"]
        curve_set["LightGBM, no privacy"] = curves["lightgbm"]
        pr = curves_frame(curve_set)
        order = list(curve_set)
        unit = alt.Scale(domain=[0, 1])
        pr_chart = alt.Chart(pr).mark_line(strokeWidth=2).encode(
            x=alt.X("recall:Q", title="Recall: share of frauds caught", scale=unit),
            y=alt.Y("precision:Q", title="Precision: alerts that are fraud", scale=unit),
            color=alt.Color("model:N", sort=order, legend=None,
                            scale=alt.Scale(domain=order, range=PALETTE["series"][:len(order)])),
            tooltip=[alt.Tooltip("model:N", title="Model"),
                     alt.Tooltip("recall:Q", title="Recall", format=".0%"),
                     alt.Tooltip("precision:Q", title="Precision", format=".0%")],
        )
        # Legend as HTML: Streamlit fits legend and plot into one fixed height, and a
        # three-row Vega legend would squash the plot.
        st.markdown(
            " &nbsp; ".join(
                f"<span style='white-space:nowrap'><span style='display:inline-block;width:14px;"
                f"height:3px;background:{color};vertical-align:middle;margin-right:6px'></span>"
                f"{name}</span>"
                for name, color in zip(order, PALETTE["series"], strict=False)
            ),
            unsafe_allow_html=True,
        )
        st.altair_chart(pr_chart.properties(height=320), use_container_width=True)

    with st.expander("All numbers"):
        table = sweep[["label", "runs", "auprc_mean", "auprc_std", "auprc_p10", "auprc_p90",
                       "share_of_no_noise", "recall_mean", "precision_mean"]]
        st.dataframe(
            table.rename(columns={
                "label": "ε", "runs": "Runs", "auprc_mean": "Mean AUPRC", "auprc_std": "Std",
                "auprc_p10": "P10", "auprc_p90": "P90", "share_of_no_noise": "Kept vs no noise",
                "recall_mean": "Recall", "precision_mean": "Precision",
            }).style.format({"Mean AUPRC": "{:.3f}", "Std": "{:.3f}", "P10": "{:.3f}",
                             "P90": "{:.3f}", "Kept vs no noise": "{:.0%}", "Recall": "{:.0%}",
                             "Precision": "{:.0%}"}),
            hide_index=True, use_container_width=True,
        )

# --- Synthetic data ----------------------------------------------------------------------------
with tab_synth:
    synthetic = summary.get("synthetic") or {}
    if not synthetic:
        st.info("This results file has no synthetic data run.")
    else:
        st.markdown(
            "Each generator learned from the real training rows and produced a same-sized fake "
            "dataset with the same fraud rate. A LightGBM model then learned **only from the fake "
            "rows** and was scored on **real** transactions it had never seen."
        )
        bars = pd.DataFrame(
            [{"source": "Real training data", "auprc": lightgbm["auprc"]}]
            + [{"source": MODEL_NAMES.get(k, k), "auprc": v["tstr"]["auprc"]}
               for k, v in synthetic.items()]
        )
        bar = alt.Chart(bars).encode(
            y=alt.Y("source:N", sort=None, title=None),
            x=alt.X("auprc:Q", scale=alt.Scale(domain=[0, 1]), title="AUPRC on real test set"),
            tooltip=[alt.Tooltip("source:N", title="Trained on"),
                     alt.Tooltip("auprc:Q", title="AUPRC", format=".3f")],
        )
        values = bar.mark_text(align="left", dx=6, fontSize=13).encode(
            text=alt.Text("auprc:Q", format=".2f")
        )
        chart = bar.mark_bar(size=24, cornerRadiusEnd=4, color=PALETTE["series"][0]) + values
        st.altair_chart(chart.properties(height=60 * len(bars) + 40), use_container_width=True)

        rows = []
        for kind, res in synthetic.items():
            rows.append({
                "Generator": MODEL_NAMES.get(kind, kind),
                "AUPRC (train synthetic, test real)": res["tstr"]["auprc"],
                "Fidelity score": res["fidelity"]["overall"],
                "Closer to training rows (all)": res["dcr_all"]["share_closer_to_train"],
                "Closer to training rows (fraud)": res["dcr_fraud"]["share_closer_to_train"],
                "Exact copies": res["dcr_all"]["exact_copies"] + res["dcr_fraud"]["exact_copies"],
            })
        st.dataframe(
            pd.DataFrame(rows).style.format({
                "AUPRC (train synthetic, test real)": "{:.3f}",
                "Fidelity score": "{:.2f}",
                "Closer to training rows (all)": "{:.0%}",
                "Closer to training rows (fraud)": "{:.0%}",
            }),
            hide_index=True, use_container_width=True,
        )
        st.markdown(
            "**Reading the memorisation check.** For each fake row, is its nearest real row one "
            "the generator trained on, or an equally sized set of real rows it never saw? Around "
            "**50%** means no sign of copying. Well above 50% means the generator is reproducing "
            "training rows. This is an empirical check, not a proof: these generators have no "
            "formal privacy guarantee, unlike the differentially private model."
        )

# --- Why ----------------------------------------------------------------------------------------
with tab_why:
    st.markdown(
        """
**Regulation.** Card transactions are personal data. In the EU, where this dataset comes from,
that means GDPR: a lawful basis for every use, purpose limitation, and fines of up to 4% of global
turnover. Card data also falls under PCI DSS. In Saudi Arabia the Personal Data Protection Law
(PDPL) and SAMA's rules for banks set the same kind of limits. None of these regimes allows
publishing raw transactions to a data science team outside the bank, a vendor, or the internet.

**Anonymising isn't enough.** A handful of amounts, times and merchants can single out one
cardholder. Even the public version of this dataset hides every feature except time and amount
behind a PCA transformation for that reason.

**Competition and security.** Fraud labels and patterns show how a bank's detection works.
Released raw, they give fraudsters a map of what gets caught and what doesn't.

So the useful question isn't "can we share it?" but "how much accuracy do we give up to learn
from it safely?" The other two tabs answer that.
"""
    )

st.caption(
    f"Results generated {summary['generated_at']} · seed {summary['config']['seed']} · "
    f"diffprivlib {summary['versions']['diffprivlib']}, SDV {summary['versions']['sdv']}, "
    f"LightGBM {summary['versions']['lightgbm']}"
)
