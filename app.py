import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "driver_data.csv"
MODEL_PATH = BASE_DIR / "churn_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.pkl"

st.set_page_config(
    page_title="Driver Churn Analytics | Portfolio",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Design system
# -----------------------------------------------------------------------------
BG = "#F7F8FB"
SURFACE = "#FFFFFF"
BORDER = "#E7E9F0"
TEXT = "#14161F"
TEXT_DIM = "#6B7080"
ACCENT = "#5B5BF6"
ACCENT_SOFT = "#EEEEFE"
RED = "#E5484D"
YELLOW = "#F5A623"
GREEN = "#12B76A"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {BG}; color: {TEXT}; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1380px; }}
    section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
    section[data-testid="stSidebar"] label {{ color: {TEXT_DIM} !important; font-size: .84rem !important; }}

    .hero {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 20px;
        padding: 1.7rem 1.8rem; margin-bottom: 1.1rem;
        box-shadow: 0 2px 8px rgba(20,22,31,.035);
    }}
    .eyebrow {{ color:{ACCENT}; font-size:.74rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin:0 0 .45rem; }}
    .title {{ font-family:'Manrope',sans-serif; font-size:2.05rem; font-weight:800; letter-spacing:-.04em; margin:0; color:{TEXT}; }}
    .subtitle {{ color:{TEXT_DIM}; font-size:.95rem; line-height:1.6; margin:.45rem 0 0; max-width:850px; }}
    .badge {{ display:inline-block; margin-top:1rem; background:{ACCENT_SOFT}; color:{ACCENT}; border-radius:999px; padding:.35rem .75rem; font-size:.72rem; font-weight:700; }}

    .section-label {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:1.08rem; margin:2rem 0 .85rem; padding-bottom:.55rem; border-bottom:1px solid {BORDER}; }}
    .card {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:1.1rem 1.25rem; box-shadow:0 1px 2px rgba(20,22,31,.035); height:100%; }}
    .card-title {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:.88rem; margin-bottom:.45rem; color:{TEXT}; }}
    .card-body {{ color:{TEXT_DIM}; font-size:.86rem; line-height:1.7; }}
    .card-body code {{ background:{ACCENT_SOFT}; color:{ACCENT}; padding:.1rem .35rem; border-radius:5px; font-weight:600; }}

    .kpi-card {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:1.15rem 1.25rem; box-shadow:0 1px 2px rgba(20,22,31,.04); }}
    .kpi-label {{ color:{TEXT_DIM}; font-size:.78rem; font-weight:600; }}
    .kpi-value {{ font-family:'Manrope',sans-serif; font-size:1.85rem; font-weight:800; line-height:1.1; margin-top:.4rem; }}
    .kpi-note {{ color:{TEXT_DIM}; font-size:.73rem; margin-top:.35rem; }}

    div[data-baseweb="select"] > div, .stTextInput input {{ border-radius:10px !important; border-color:{BORDER} !important; }}
    .stButton>button {{ border-radius:10px; font-weight:600; }}
    [data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:14px; overflow:hidden; }}
    [data-testid="stPlotlyChart"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:.7rem; box-shadow:0 1px 2px rgba(20,22,31,.035); }}
    [data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; padding:.8rem 1rem; }}
    .small-note {{ color:{TEXT_DIM}; font-size:.78rem; line-height:1.6; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, color=TEXT_DIM),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, l=10, r=10, b=10),
    )
)
RISK_COLORS = {"High Risk": RED, "Medium Risk": YELLOW, "Low Risk": GREEN}


def style_fig(fig):
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def kpi(label, value, note):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data

def load_metrics():
    with open(METRICS_PATH, "rb") as f:
        return pickle.load(f)


def engineer_features(df):
    """Feature engineering aligned with the training notebook."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.strip()
    # Driver ID must be text and contain at least 6 characters
    df["driverid"] = df["driverid"].astype(str).str.strip()

    # Remove rows with Driver IDs shorter than 6 characters
    df = df[df["driverid"].str.len() >= 5].copy()
    current_date = pd.Timestamp("2025-12-24")
    df["signupdate"] = pd.to_datetime(df["signupdate"], errors="coerce")
    df["lastactivedate"] = pd.to_datetime(df["lastactivedate"], errors="coerce").fillna(current_date)
    df["activeduration_days"] = (df["lastactivedate"] - df["signupdate"]).dt.days.clip(lower=0)
    df["inactivity_days"] = (current_date - df["lastactivedate"]).dt.days.clip(lower=0)

    missing_cols = ["ratings", "averagedailyearnings_ghs", "utilisation", "tripsperweek",
                    "hoursonlineperday", "earningsperweek", "cancellationrate", "engagementscore"]
    for col in missing_cols:
        if col in df.columns:
            df[f"missing_{col}"] = df[col].isna().astype(int)

    numeric_cols = ["dayssincelasttrip", "totaltrips", "tripsperweek", "hoursonlineperday",
                    "utilisation", "averagedailyearnings_ghs", "earningsperweek", "ratings",
                    "cancellationrate", "incentiveparticipation", "engagementscore"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["trips_per_day"] = df["totaltrips"] / df["activeduration_days"].replace(0, np.nan)
    df["daily_trips"] = df["tripsperweek"] / 7
    df["earnings_per_trip"] = df["averagedailyearnings_ghs"] / df["daily_trips"].replace(0, np.nan)
    df["online_efficiency"] = df["utilisation"] * df["hoursonlineperday"]
    df["recent_activity_flag"] = (df["inactivity_days"] > 30).astype(int)
    df["cancellation_bin"] = pd.cut(df["cancellationrate"], [0, .2, .35, 1], labels=["Low", "Medium", "High"], include_lowest=True)
    df["earnings_to_trip_ratio"] = df["earningsperweek"] / df["tripsperweek"].replace(0, np.nan)
    df["incentive_engagement"] = df["incentiveparticipation"] * df["engagementscore"]
    df["active_hour_pattern"] = pd.cut(df["hoursonlineperday"], [0, 6, 12, 24], labels=["Part-time", "Full-time", "Over-time"], include_lowest=True)
    df["engagement_x_trips"] = df["engagementscore"] * df["tripsperweek"]
    df["utilisation_x_hours"] = df["utilisation"] * df["hoursonlineperday"]
    df["earnings_x_engagement"] = df["averagedailyearnings_ghs"] * df["engagementscore"]
    df["new_driver_flag"] = (df["activeduration_days"] < 90).astype(int)

    med = {c: df[c].median() for c in ["totaltrips", "earningsperweek", "cancellationrate", "inactivity_days", "engagementscore"]}
    df["high_value_driver"] = ((df["totaltrips"] > med["totaltrips"]) & (df["earningsperweek"] > med["earningsperweek"]) & (df["cancellationrate"] < med["cancellationrate"])).astype(int)
    df["at_risk_driver"] = ((df["inactivity_days"] > med["inactivity_days"]) & (df["engagementscore"] < med["engagementscore"]) & (df["cancellationrate"] > med["cancellationrate"])).astype(int)
    return df


@st.cache_data

def score_drivers():
    df = engineer_features(pd.read_csv(DATA_PATH))
    bundle = load_model()
    model = bundle["pipeline"]
    features = bundle["features"]
    X = df[features].copy()
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["predicted_churn"] = model.predict(X)
    df["risk_category"] = pd.cut(df["churn_probability"], [-np.inf, .30, .70, np.inf], labels=["Low Risk", "Medium Risk", "High Risk"])
    return df, bundle


def section(title):
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)


def main():
    # Hero / case-study framing
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Customer Analytics</div>
            <div class="title">Driver Churn Prediction</div>
            <div class="subtitle">
                An end-to-end machine learning dashboard that identifies drivers at risk of churn,
                quantifies retention risk, and translates model output into actionable fleet insights.
            </div>
            <div class="badge">Python · Scikit-learn · Streamlit · Plotly · Logistic Regression</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        st.error("Required project files are missing. Ensure driver_Churn.csv and churn_model.pkl are in the project folder.")
        return

    with st.spinner("Scoring driver fleet..."):
        df, bundle = score_drivers()
        metrics = load_metrics() if METRICS_PATH.exists() else None

    # Portfolio summary
    section("Project at a glance")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="card"><div class="card-title">Business question</div><div class="card-body">Which drivers show the strongest signals of churn, and where should retention activity be prioritised?</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-title">Analytical approach</div><div class="card-body">Feature engineering → preprocessing → balanced Logistic Regression → probability-based risk segmentation.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card"><div class="card-title">Decision output</div><div class="card-body">A ranked, filterable driver view that converts churn probabilities into Low, Medium and High Risk segments.</div></div>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("## Dashboard controls")
    cities = sorted(df["city"].dropna().astype(str).unique())
    selected_cities = st.sidebar.multiselect("City", cities, default=cities)
    risks = st.sidebar.multiselect("Risk category", ["High Risk", "Medium Risk", "Low Risk"], default=["High Risk", "Medium Risk", "Low Risk"])
    min_inactivity = st.sidebar.slider("Minimum inactivity (days)", 0, max(30, int(df["inactivity_days"].max())), 0)

    filtered = df[df["city"].isin(selected_cities) & df["risk_category"].isin(risks) & (df["inactivity_days"] >= min_inactivity)].copy()
    if filtered.empty:
        st.warning("No drivers match the current filters.")
        return

    section("Fleet risk overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Drivers scored", f"{len(filtered):,}", f"of {len(df):,} total records")
    with c2: kpi("Predicted churn", f"{filtered.predicted_churn.mean():.1%}", f"{filtered.predicted_churn.sum():,} predicted churn cases")
    with c3: kpi("High-risk drivers", f"{(filtered.risk_category == 'High Risk').sum():,}", "probability > 70%")
    with c4: kpi("Avg. churn probability", f"{filtered.churn_probability.mean():.1%}", "model-estimated fleet risk")

    section("Model performance")
    if metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: kpi("ROC-AUC", f"{metrics['roc_auc']:.3f}", "holdout test set")
        with m2: kpi("Accuracy", f"{metrics['accuracy']:.1%}", "holdout test set")
        with m3: kpi("Precision", f"{metrics['precision']:.1%}", "churn class")
        with m4: kpi("Recall", f"{metrics['recall']:.1%}", "churn class")
        with m5: kpi("CV ROC-AUC", f"{metrics['cv_roc_auc_mean']:.3f}", f"± {metrics['cv_roc_auc_std']:.3f}")

    st.markdown('<p class="small-note">The model uses a balanced Logistic Regression pipeline with engineered behavioural, activity, engagement and interaction features. Risk thresholds are applied to predicted churn probability for prioritisation.</p>', unsafe_allow_html=True)

    section("Risk landscape")
    left, right = st.columns(2)
    with left:
        counts = filtered["risk_category"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0).reset_index()
        counts.columns = ["Risk", "Drivers"]
        fig = px.bar(counts, x="Risk", y="Drivers", color="Risk", color_discrete_map=RISK_COLORS, title="Drivers by risk segment")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        fig = px.histogram(filtered, x="churn_probability", nbins=30, title="Predicted churn probability")
        fig.add_vline(x=.30, line_dash="dash", annotation_text="30%")
        fig.add_vline(x=.70, line_dash="dash", annotation_text="70%")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    section("What drives churn risk?")
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]
    imp = pd.DataFrame({"feature": names, "coefficient": coefs})
    imp["feature"] = imp["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
    imp["abs"] = imp["coefficient"].abs()
    top = imp.nlargest(12, "abs").sort_values("coefficient")
    top["direction"] = np.where(top["coefficient"] >= 0, "Higher churn odds", "Lower churn odds")
    fig = px.bar(top, x="coefficient", y="feature", color="direction", orientation="h", title="Top Logistic Regression coefficients")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("Positive coefficients are associated with higher estimated churn odds; negative coefficients are associated with lower estimated churn odds. This is model association, not causal inference.")

    section("Behavioural signals")
    feature_options = ["inactivity_days", "trips_per_day", "earnings_per_trip", "online_efficiency", "engagementscore", "cancellationrate", "incentive_engagement"]
    feature = st.selectbox("Explore a feature", feature_options)
    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtered, x=feature, color="risk_category", nbins=35, color_discrete_map=RISK_COLORS, title=f"{feature} across risk groups")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        sample = filtered.sample(min(750, len(filtered)), random_state=42)
        fig = px.scatter(sample, x=feature, y="churn_probability", color="risk_category", color_discrete_map=RISK_COLORS, hover_data=["driverid", "city", "inactivity_days"], title=f"{feature} vs predicted churn probability")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    section("Priority driver list")
    search = st.text_input("Search by Driver ID", placeholder="e.g. DRV...")
    roster = filtered.copy()
    if search:
        roster = roster[roster["driverid"].astype(str).str.contains(search, case=False, na=False)]
    roster = roster.sort_values("churn_probability", ascending=False)
    cols = ["driverid", "city", "inactivity_days", "tripsperweek", "engagementscore", "cancellationrate", "churn_probability", "risk_category"]
    shown = roster[cols].head(100).copy()
    shown["churn_probability"] = shown["churn_probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(shown, use_container_width=True, height=430)

    export = filtered.copy()
    st.download_button("Download scored driver results", export.to_csv(index=False), file_name="driver_churn_scored_results.csv", mime="text/csv")

    section("Methodology")
    st.markdown(
        """
        <div class="card">
        <div class="card-body">
        <b>Target:</b> driver churn classification.<br>
        <b>Model:</b> Logistic Regression with class balancing.<br>
        <b>Preprocessing:</b> median imputation and standardisation for numeric variables; most-frequent imputation and one-hot encoding for categorical variables.<br>
        <b>Feature engineering:</b> inactivity, active duration, trip intensity, earnings efficiency, utilisation, engagement interactions, cancellation bands, missingness indicators and driver lifecycle flags.<br>
        <b>Validation:</b> stratified holdout evaluation plus 5-fold cross-validation using ROC-AUC.<br>
        <b>Portfolio purpose:</b> demonstrate the full path from business problem and feature engineering to model evaluation and an operational decision interface.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(f'<p class="small-note">project · Driver Churn Analytics · Last refreshed {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
