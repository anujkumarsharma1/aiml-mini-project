"""
AquaGuard AI – Water Quality Classification System
Streamlit Web Application

A clean, professional, and beginner-friendly web application for classifying
water quality using machine learning, built with Streamlit, Scikit-learn,
Pandas, and Seaborn.
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AquaGuard AI – Water Quality Classification",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Water-Themed CSS Styling
st.markdown(
    """
    <style>
    /* Main container and typography */
    .main-title {
        color: #0369a1;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        color: #0284c7;
    }
    .metric-card p {
        margin: 0;
        font-size: 0.9rem;
        color: #334155;
        font-weight: 500;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        color: #92400e;
        font-size: 0.92rem;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .badge-model {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .pred-potable {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
    }
    .pred-non-potable {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Data & Model Loading Helper Functions (Cached for Speed)
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Load genuine water potability dataset."""
    csv_file = "water_potability.csv"
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return None


@st.cache_data
def load_results():
    """Load precomputed training and evaluation metrics from model_results.json."""
    json_file = "model_results.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_trained_model():
    """Load serialized Scikit-learn Pipeline (SimpleImputer + StandardScaler + Model)."""
    model_file = "water_model.joblib"
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return None


def metrics_row(tag, m):
    """One row of the comparison table from a metrics dictionary."""
    return {
        "Stage": tag,
        "Model": m["name"],
        "Accuracy": f"{m['accuracy'] * 100:.2f}%",
        "Precision": f"{m['precision']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "F1-Score": f"{m['f1_score']:.4f}",
        "ROC-AUC": f"{m['roc_auc']:.4f}",
    }


def show_confusion(title, m, cmap):
    """Draw one confusion-matrix heatmap with TN/FP/FN/TP caption."""
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 3.6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
        xticklabels=["Pred Non-Potable (0)", "Pred Potable (1)"],
        yticklabels=["Actual Non-Potable (0)", "Actual Potable (1)"],
    )
    ax.set_title(f"{title}\n(Accuracy {m['accuracy']*100:.2f}%, Recall {m['recall']:.2f})", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    tn, fp, fn, tp = cm.ravel()
    st.caption(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")


@st.cache_data
def scaled_features(data):
    """Median-impute and standardise the 9 features (labels are NOT used) for unsupervised analysis."""
    features = data.drop(columns=["Potability"])
    filled = SimpleImputer(strategy="median").fit_transform(features)
    return StandardScaler().fit_transform(filled)


@st.cache_data
def run_pca(data):
    """Fit PCA with all components; return 2-D coordinates and explained-variance ratios."""
    pca = PCA(random_state=42).fit(scaled_features(data))
    coords = pca.transform(scaled_features(data))[:, :2]
    return coords, pca.explained_variance_ratio_


@st.cache_data
def kmeans_sweep(data, k_max=10):
    """Inertia (for the elbow plot) and silhouette score for k = 2..k_max."""
    X_scaled = scaled_features(data)
    inertias, silhouettes = {}, {}
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        inertias[k] = float(km.inertia_)
        silhouettes[k] = float(silhouette_score(X_scaled, km.labels_))
    # k = 1 inertia is the total variance, needed for the first point of the elbow curve
    inertias[1] = float(((X_scaled - X_scaled.mean(axis=0)) ** 2).sum())
    return dict(sorted(inertias.items())), silhouettes


def cluster_agreement(clusters, labels):
    """Compare cluster ids with the real potability labels: cross-tab, purity and Adjusted Rand Index."""
    table = pd.crosstab(
        pd.Series(clusters, name="Cluster"),
        pd.Series(labels, name="Actual").map({0: "Non-Potable (0)", 1: "Potable (1)"}),
    )
    purity = float(table.max(axis=1).sum() / table.values.sum())
    ari = float(adjusted_rand_score(labels, clusters))
    return table, purity, ari


df = load_dataset()
results = load_results()
model_pipeline = load_trained_model()

# Verify that essential artifacts are available
if df is None or results is None or model_pipeline is None:
    st.error(
        "⚠️ Project artifacts not found! Please run `python train_model.py` in your terminal "
        "first to generate `water_model.joblib` and `model_results.json`."
    )
    st.stop()

feature_names = results["dataset_info"]["features"]
feature_stats = results.get("feature_statistics", {})
presets = results.get("presentation_presets", {})


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Overview
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1548839140-29a749e1bc4e?auto=format&fit=crop&w=400&q=80",
        caption="AquaGuard AI System",
        use_container_width=True,
    )
    st.markdown("## 💧 AquaGuard AI")
    st.markdown(
        "**Water Quality Classification System**\n\n"
        "A practical Machine Learning project utilizing real-world water physicochemical parameters."
    )
    st.markdown("---")

    navigation = st.radio(
        "📌 Navigation Menu",
        [
            "💧 Water Quality Predictor",
            "📊 Dataset Analysis",
            "🔬 Model Performance & Comparison",
            "🧩 Unsupervised Analysis",
            "📖 Academic Report & Viva Guide",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.success("✅ Dataset Loaded (3,276 records)")
    st.success("✅ Model Pipeline Ready")
    st.caption(f"Algorithm: {results['selected_model']['name']}")
    st.caption("College Project Edition • 100% Genuine ML Logic")


# -----------------------------------------------------------------------------
# TAB 1: Water Quality Prediction Interface
# -----------------------------------------------------------------------------
if navigation == "💧 Water Quality Predictor":
    st.markdown('<div class="main-title">💧 AquaGuard AI – Water Quality Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Input water physical and chemical parameters to estimate potability '
        'based on patterns learned from historical dataset records.</div>',
        unsafe_allow_html=True,
    )

    # Presentation helper: Preset loaders
    st.markdown("##### 🧪 Quick Presets (Ideal for College Live Demonstrations)")
    preset_col1, preset_col2, preset_col3 = st.columns(3)

    # Initialize session state for inputs if not present
    default_vals = {
        feat: feature_stats[feat]["median"] for feat in feature_names if feat in feature_stats
    }
    for feat in feature_names:
        if feat not in st.session_state:
            st.session_state[feat] = float(default_vals.get(feat, 7.0))

    if preset_col1.button("📌 Load Known Potable Sample", use_container_width=True):
        if "potable_sample" in presets:
            for feat, val in presets["potable_sample"].items():
                st.session_state[feat] = float(val)
            st.rerun()

    if preset_col2.button("📌 Load Known Non-Potable Sample", use_container_width=True):
        if "non_potable_sample" in presets:
            for feat, val in presets["non_potable_sample"].items():
                st.session_state[feat] = float(val)
            st.rerun()

    if preset_col3.button("🔄 Reset to Dataset Medians", use_container_width=True):
        for feat in feature_names:
            st.session_state[feat] = float(default_vals.get(feat, 7.0))
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎛️ Enter Water Quality Parameters")

    # Group the 9 features logically into 3 columns
    input_data = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🧪 Acid-Base & Disinfection")
        input_data["ph"] = st.number_input(
            "1. pH Value",
            min_value=0.0,
            max_value=14.0,
            value=float(st.session_state.get("ph", 7.0)),
            step=0.1,
            help="WHO standard: 6.5 – 8.5. Measures acidity/alkalinity of water.",
        )
        input_data["Chloramines"] = st.number_input(
            "4. Chloramines (ppm)",
            min_value=0.0,
            max_value=20.0,
            value=float(st.session_state.get("Chloramines", 7.13)),
            step=0.1,
            help="Disinfectant derived from chlorine and ammonia. Safe limits < 4 ppm.",
        )
        input_data["Organic_carbon"] = st.number_input(
            "7. Organic Carbon (ppm)",
            min_value=0.0,
            max_value=40.0,
            value=float(st.session_state.get("Organic_carbon", 14.22)),
            step=0.1,
            help="Total Organic Carbon (TOC) from decaying natural matter. EPA limit < 2-4 ppm.",
        )

    with col2:
        st.markdown("##### 🪨 Dissolved Minerals & Solids")
        input_data["Hardness"] = st.number_input(
            "2. Hardness (mg/L)",
            min_value=0.0,
            max_value=400.0,
            value=float(st.session_state.get("Hardness", 196.97)),
            step=1.0,
            help="Capacity of water to precipitate soap; caused by calcium and magnesium.",
        )
        input_data["Sulfate"] = st.number_input(
            "5. Sulfate (mg/L)",
            min_value=0.0,
            max_value=600.0,
            value=float(st.session_state.get("Sulfate", 333.07)),
            step=1.0,
            help="Naturally occurring mineral salts. Recommended standard < 250 mg/L.",
        )
        input_data["Trihalomethanes"] = st.number_input(
            "8. Trihalomethanes (μg/L)",
            min_value=0.0,
            max_value=150.0,
            value=float(st.session_state.get("Trihalomethanes", 66.62)),
            step=0.5,
            help="By-products formed when chlorine reacts with organic matter. Safe < 80 μg/L.",
        )

    with col3:
        st.markdown("##### ⚡ Physical Properties & Clarity")
        input_data["Solids"] = st.number_input(
            "3. Solids / TDS (ppm)",
            min_value=0.0,
            max_value=70000.0,
            value=float(st.session_state.get("Solids", 20927.83)),
            step=100.0,
            help="Total Dissolved Solids (TDS). Desirable standard < 500-1000 ppm.",
        )
        input_data["Conductivity"] = st.number_input(
            "6. Conductivity (μS/cm)",
            min_value=0.0,
            max_value=1000.0,
            value=float(st.session_state.get("Conductivity", 421.88)),
            step=1.0,
            help="Electrical conductivity influenced by dissolved ions. WHO standard < 400 μS/cm.",
        )
        input_data["Turbidity"] = st.number_input(
            "9. Turbidity (NTU)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.get("Turbidity", 3.96)),
            step=0.1,
            help="Measure of water clarity / cloudiness from suspended particles. WHO limit < 5 NTU.",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🚀 Predict Water Quality", type="primary", use_container_width=True)

    if predict_button:
        # Build single-sample DataFrame with matching feature names
        sample_df = pd.DataFrame([input_data])[feature_names]

        # Model Prediction using the saved pipeline
        prediction = int(model_pipeline.predict(sample_df)[0])
        probabilities = model_pipeline.predict_proba(sample_df)[0]
        prob_non_potable = float(probabilities[0] * 100)
        prob_potable = float(probabilities[1] * 100)

        model_name = results["selected_model"]["name"]

        st.markdown("### 📋 Prediction Results")
        st.markdown(f'<span class="badge-model">Generated by: {model_name}</span>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(
                f"""
                <div class="pred-potable">
                    <h3 style="margin:0 0 8px 0; color:#065f46;">✅ Predicted Class: 1 – Potable</h3>
                    <p style="margin:0; font-size:1.05rem;">
                        According to the historical dataset labels, this water sample exhibits patterns consistent with 
                        <strong>Potable Water (Class 1)</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="pred-non-potable">
                    <h3 style="margin:0 0 8px 0; color:#991b1b;">⚠️ Predicted Class: 0 – Non-Potable</h3>
                    <p style="margin:0; font-size:1.05rem;">
                        According to the historical dataset labels, this water sample exhibits patterns consistent with 
                        <strong>Non-Potable Water (Class 0)</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        # Probability Breakdown
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Estimated Potability Probability (Class 1)", f"{prob_potable:.1f}%")
            st.progress(prob_potable / 100.0)
        with prob_col2:
            st.metric("Estimated Non-Potability Probability (Class 0)", f"{prob_non_potable:.1f}%")
            st.progress(prob_non_potable / 100.0)

    # Mandatory Academic Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Academic & Safety Disclaimer:</strong><br>
            "AquaGuard AI is an educational machine learning project. Its predictions reflect patterns learned 
            from historical dataset labels and do not establish whether water is safe to drink. Laboratory testing 
            and applicable drinking-water standards are required to assess actual water safety."
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# TAB 2: Dataset Analysis
# -----------------------------------------------------------------------------
elif navigation == "📊 Dataset Analysis":
    st.markdown('<div class="main-title">📊 Dataset Analysis & Exploration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Real statistical breakdown of the verified Kaggle Water Potability Dataset.</div>',
        unsafe_allow_html=True,
    )

    info = results["dataset_info"]

    # Key Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f'<div class="metric-card"><h3>{info["total_records"]}</h3><p>Total Records</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><h3>{info["feature_count"]}</h3><p>Input Features</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><h3>{info["class_distribution"]["Potable (1)"]}</h3><p>Potable (Class 1)</p></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><h3>{info["class_distribution"]["Non-Potable (0)"]}</h3><p>Non-Potable (Class 0)</p></div>',
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f'<div class="metric-card"><h3>{info["duplicate_records"]}</h3><p>Duplicates Found</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Missing values & Class Distribution Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 1. Target Class Distribution")
        fig_class, ax_class = plt.subplots(figsize=(6, 4))
        classes = ["Non-Potable (0)", "Potable (1)"]
        counts = [
            info["class_distribution"]["Non-Potable (0)"],
            info["class_distribution"]["Potable (1)"],
        ]
        colors = ["#f87171", "#34d399"]
        bars = ax_class.bar(classes, counts, color=colors, width=0.55, edgecolor="#334155")
        ax_class.set_ylabel("Number of Samples", fontsize=10)
        ax_class.set_title("Distribution of Potable vs Non-Potable Samples", fontsize=11, fontweight="bold")
        for bar in bars:
            height = bar.get_height()
            percentage = (height / sum(counts)) * 100
            ax_class.annotate(
                f"{height} ({percentage:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax_class.set_ylim(0, max(counts) * 1.18)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_class)
        st.caption("The dataset contains 61.0% Non-Potable and 39.0% Potable records, reflecting slight class imbalance.")

    with chart_col2:
        st.markdown("#### 2. Missing Values by Feature")
        missing_series = pd.Series(info["missing_values"])
        missing_filtered = missing_series[missing_series > 0].sort_values(ascending=False)

        fig_miss, ax_miss = plt.subplots(figsize=(6, 4))
        miss_bars = ax_miss.bar(
            missing_filtered.index,
            missing_filtered.values,
            color="#38bdf8",
            width=0.5,
            edgecolor="#0284c7",
        )
        ax_miss.set_ylabel("Missing Count", fontsize=10)
        ax_miss.set_title("Features with Missing Values (Handled via Median Imputation)", fontsize=11, fontweight="bold")
        for bar in miss_bars:
            height = bar.get_height()
            pct = (height / info["total_records"]) * 100
            ax_miss.annotate(
                f"{int(height)} ({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax_miss.set_ylim(0, max(missing_filtered.values) * 1.18)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_miss)
        st.caption("Only Sulfate (781), pH (491), and Trihalomethanes (162) contain missing values.")

    st.markdown("---")

    # Correlation Heatmap
    st.markdown("#### 3. Correlation Heatmap Between Physicochemical Features")
    fig_corr, ax_corr = plt.subplots(figsize=(9, 5.5))
    corr_matrix = df.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        ax=ax_corr,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    plt.title("Pearson Correlation Heatmap (Water Quality Features)", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    st.pyplot(fig_corr)
    st.info(
        "💡 **Key Insight for Presentation:** The correlation coefficients between individual features and the target "
        "`Potability` are nearly zero (ranging from -0.03 to +0.03). This explains why a linear classifier like "
        "Logistic Regression has almost no predictive power (ROC-AUC about 0.55). Non-linear models (Random Forest, SVM, "
        "Decision Tree) do somewhat better, but no model gets far above 0.66 ROC-AUC, so each feature carries little "
        "information by itself."
    )

    st.markdown("---")
    st.markdown("#### 4. Sample Genuine Dataset Records")
    st.dataframe(df.head(10), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Model Performance & Comparison
# -----------------------------------------------------------------------------
elif navigation == "🔬 Model Performance & Comparison":
    st.markdown('<div class="main-title">🔬 Model Performance & Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Imbalance-aware retraining: original models vs class-weighted, F1-tuned models '
        '(all numbers come from <code>train_model.py</code> and are evaluated on the same held-out test set).</div>',
        unsafe_allow_html=True,
    )

    test_size = results["dataset_info"]["test_size"]
    original_models = results["models"]
    improved_models = results["improved_models"]
    tuning = results["tuning"]

    # ---- 1. Syllabus model bake-off -------------------------------------------
    st.markdown(f"### 1. Model Bake-off: 7 Classifiers on the Same Split (Test Set: {test_size} Samples)")
    st.caption(
        f"Same preprocessing pipeline, same train/test split. Every model was tuned with GridSearchCV "
        f"(scoring = **{tuning['scoring'].upper()}**, {tuning['cv']}). "
        "KNN, Naive Bayes and AdaBoost do not support `class_weight` in scikit-learn."
    )
    bake_rows = []
    for key, m in improved_models.items():
        bake_rows.append({
            "Model": m["name"],
            "CV F1 (train)": f"{m['cv_f1']:.4f}",
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "class_weight": m["class_weight"],
            "Best parameters": ", ".join(f"{k}={v}" for k, v in m["best_parameters"].items()) or "-",
        })
    bake_df = pd.DataFrame(bake_rows).sort_values("CV F1 (train)", ascending=False).reset_index(drop=True)
    st.table(bake_df)

    best_f1_key = max(improved_models, key=lambda k: improved_models[k]["f1_score"])
    best_auc_key = max(improved_models, key=lambda k: improved_models[k]["roc_auc"])
    st.caption(
        f"Selected by CV F1: **{results['selected_model']['name']}**. "
        f"Highest test F1: {improved_models[best_f1_key]['name']} ({improved_models[best_f1_key]['f1_score']:.4f}). "
        f"Highest test ROC-AUC: {improved_models[best_auc_key]['name']} ({improved_models[best_auc_key]['roc_auc']:.4f}). "
        "The differences between the top models are small, so no single model is clearly better on this dataset."
    )

    # F1 and ROC-AUC bar charts
    short_names = [m["name"].split(" (")[0] for m in improved_models.values()]
    bake_col1, bake_col2 = st.columns(2)
    for col, metric_key, title, color in [
        (bake_col1, "f1_score", "Test F1-Score", "#4f46e5"),
        (bake_col2, "roc_auc", "Test ROC-AUC", "#0284c7"),
    ]:
        with col:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            values = [m[metric_key] for m in improved_models.values()]
            bars = ax_bar.barh(short_names, values, color=color)
            ax_bar.set_xlim(0, 1)
            ax_bar.invert_yaxis()
            ax_bar.set_title(title, fontweight="bold")
            for bar, val in zip(bars, values):
                ax_bar.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
            if metric_key == "roc_auc":
                ax_bar.axvline(0.5, color="#94a3b8", linestyle="--", linewidth=1)
                ax_bar.text(0.51, len(values) - 0.45, "random = 0.5", fontsize=8, color="#64748b")
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

    # Confusion matrices of all 7 models, three per row
    st.markdown("#### Confusion Matrices of All Models")
    model_items = list(improved_models.items())
    for start in range(0, len(model_items), 3):
        cm_cols = st.columns(3)
        for col, (key, m) in zip(cm_cols, model_items[start:start + 3]):
            with col:
                show_confusion(m["name"].split(" (")[0], m, "Blues")

    st.markdown("---")
    # ---- 2. Before vs After table -------------------------------------------
    st.markdown(f"### 2. Before vs After (Held-Out Test Set: {test_size} Samples)")
    st.caption(
        f"**Before** = original approach (no class weights, tuned for accuracy). "
        f"**After** = `class_weight='balanced'` where supported, GridSearchCV scoring = **{tuning['scoring'].upper()}**, "
        f"cv = {tuning['cv']}."
    )
    rows = []
    for tag, m in [
        ("Before", original_models["logistic_regression"]),
        ("Before", original_models["random_forest_baseline"]),
        ("Before", original_models["random_forest_optimized"]),
        ("After", improved_models["logistic_regression"]),
        ("After", improved_models["random_forest"]),
    ]:
        rows.append(metrics_row(tag, m))
    st.table(pd.DataFrame(rows))

    # ---- 3. Grouped bar chart: RF before vs after -----------------------------
    st.markdown("---")
    st.markdown("#### 3. Random Forest: Before vs After")
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rf_before = original_models["random_forest_optimized"]
    rf_after = improved_models["random_forest"]

    fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(metric_keys))
    width = 0.38
    bars_b = ax_cmp.bar(x_pos - width / 2, [rf_before[k] for k in metric_keys], width, label="Before (accuracy-tuned)", color="#94a3b8")
    bars_a = ax_cmp.bar(x_pos + width / 2, [rf_after[k] for k in metric_keys], width, label="After (balanced + F1-tuned)", color="#0284c7")
    for bar in list(bars_b) + list(bars_a):
        ax_cmp.annotate(f"{bar.get_height():.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax_cmp.set_xticks(x_pos)
    ax_cmp.set_xticklabels(metric_labels)
    ax_cmp.set_ylim(0, 1)
    ax_cmp.legend()
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_cmp)
    st.caption(
        f"Recall for potable water rises from {rf_before['recall']:.2f} to {rf_after['recall']:.2f} and F1 from "
        f"{rf_before['f1_score']:.2f} to {rf_after['f1_score']:.2f}, but accuracy drops from "
        f"{rf_before['accuracy']*100:.1f}% to {rf_after['accuracy']*100:.1f}%. "
        "This is the usual trade-off: the model now catches more potable samples at the cost of more false alarms."
    )

    # ---- 4. Confusion matrices --------------------------------------------------
    st.markdown("---")
    st.markdown(f"#### 4. Before vs After Confusion Matrices (Held-Out Test Set: {test_size} Samples)")
    cm_row1 = st.columns(2)
    with cm_row1[0]:
        show_confusion("Logistic Regression: Before", original_models["logistic_regression"], "Blues")
    with cm_row1[1]:
        show_confusion("Logistic Regression: After (balanced)", improved_models["logistic_regression"], "Blues")
    cm_row2 = st.columns(2)
    with cm_row2[0]:
        show_confusion("Random Forest: Before (accuracy-tuned)", rf_before, "Greens")
    with cm_row2[1]:
        show_confusion("Random Forest: After (balanced, F1-tuned)", rf_after, "Greens")

    # ---- 5. Feature importances ---------------------------------------------------
    st.markdown("---")
    selected = results["selected_model"]
    st.markdown(f"#### 5. Feature Importances (Selected Model: {selected['name']})")
    feat_imps = selected["feature_importances"]
    feat_df = pd.DataFrame({"Feature": list(feat_imps.keys()), "Importance": list(feat_imps.values())})

    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="Blues_r", ax=ax_imp)
    ax_imp.set_title("Relative Predictive Power of Water Quality Parameters", fontweight="bold")
    ax_imp.set_xlabel("Importance (drop in F1 when the feature is shuffled)")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_imp)
    top3 = ", ".join(list(feat_imps.keys())[:3])
    st.caption(f"Method: {selected['importance_method']}. Top 3 features: {top3}.")
    st.info(f"**Why this model?** {selected['selection_criterion']}")

    st.markdown("---")
    st.markdown("#### 6. Educational Metrics Explanation (Viva Ready)")
    exp1, exp2, exp3, exp4, exp5 = st.columns(5)
    with exp1:
        st.markdown("**Accuracy**")
        st.caption("Overall proportion of correct predictions: `(TP + TN) / Total`. Can be misleading on imbalanced datasets.")
    with exp2:
        st.markdown("**Precision**")
        st.caption("Of all samples predicted as Potable, how many were truly potable: `TP / (TP + FP)`.")
    with exp3:
        st.markdown("**Recall (Sensitivity)**")
        st.caption("Of all actual potable samples, how many did the model find: `TP / (TP + FN)`.")
    with exp4:
        st.markdown("**F1-Score**")
        st.caption("Harmonic mean of Precision and Recall: `2 * (P * R) / (P + R)`. Better than accuracy when classes are imbalanced.")
    with exp5:
        st.markdown("**ROC-AUC**")
        st.caption("Chance that a random potable sample gets a higher score than a random non-potable one. 0.5 = random guessing, 1.0 = perfect.")


# -----------------------------------------------------------------------------
# TAB: Unsupervised Analysis (PCA, K-Means, Hierarchical Clustering)
# -----------------------------------------------------------------------------
elif navigation == "🧩 Unsupervised Analysis":
    st.markdown('<div class="main-title">🧩 Unsupervised Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">PCA, K-Means and hierarchical clustering on the 9 water-quality features. '
        'The labels are <b>not</b> used to build the clusters; they are only used afterwards to check '
        'whether the clusters match potability.</div>',
        unsafe_allow_html=True,
    )

    y_all = df["Potability"].to_numpy()
    majority_share = max(y_all.mean(), 1 - y_all.mean())
    st.caption(
        "Preprocessing: median imputation + StandardScaler on all 3,276 rows (features only). "
        f"Majority-class share = {majority_share * 100:.1f}%, used below as the 'always guess one class' baseline."
    )

    # ---- 1. PCA ---------------------------------------------------------------
    st.markdown("### 1. PCA: Do Potable and Non-Potable Samples Separate?")
    coords, var_ratio = run_pca(df)
    pca_col1, pca_col2 = st.columns(2)

    with pca_col1:
        fig_pca, ax_pca = plt.subplots(figsize=(6, 4.5))
        for label, color, name in [(0, "#f87171", "Non-Potable (0)"), (1, "#10b981", "Potable (1)")]:
            mask = y_all == label
            ax_pca.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.4, color=color, label=name)
        ax_pca.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}% variance)")
        ax_pca.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}% variance)")
        ax_pca.set_title("PCA Projection to 2-D", fontweight="bold")
        ax_pca.legend()
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_pca)
        plt.close(fig_pca)

    with pca_col2:
        cumulative = np.cumsum(var_ratio)
        components = np.arange(1, len(var_ratio) + 1)
        fig_var, ax_var = plt.subplots(figsize=(6, 4.5))
        ax_var.bar(components, var_ratio * 100, color="#38bdf8", label="Per component")
        ax_var.plot(components, cumulative * 100, color="#0f172a", marker="o", label="Cumulative")
        ax_var.set_xlabel("Principal component")
        ax_var.set_ylabel("Explained variance (%)")
        ax_var.set_xticks(components)
        ax_var.set_title("Explained Variance", fontweight="bold")
        ax_var.legend()
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_var)
        plt.close(fig_var)

    n_for_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    st.info(
        f"The first 2 components explain only **{cumulative[1] * 100:.1f}%** of the variance, and "
        f"{n_for_90} of 9 components are needed to reach 90%. The two colours overlap almost completely, "
        "so potable and non-potable samples cannot be separated by simple structure in the features. "
        "This fits the modest accuracy (about 60-66%) of every classifier in the bake-off."
    )

    # ---- 2. K-Means ---------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 2. K-Means: Elbow Method and Silhouette Score")
    inertias, silhouettes = kmeans_sweep(df)
    km_col1, km_col2 = st.columns(2)

    with km_col1:
        fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
        ax_elbow.plot(list(inertias.keys()), list(inertias.values()), marker="o", color="#0284c7")
        ax_elbow.set_xlabel("Number of clusters (k)")
        ax_elbow.set_ylabel("Inertia (within-cluster sum of squares)")
        ax_elbow.set_title("Elbow Method", fontweight="bold")
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_elbow)
        plt.close(fig_elbow)

    with km_col2:
        fig_sil, ax_sil = plt.subplots(figsize=(6, 4))
        ax_sil.plot(list(silhouettes.keys()), list(silhouettes.values()), marker="o", color="#4f46e5")
        ax_sil.set_xlabel("Number of clusters (k)")
        ax_sil.set_ylabel("Silhouette score")
        ax_sil.set_title("Silhouette Score vs k", fontweight="bold")
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_sil)
        plt.close(fig_sil)

    best_k = max(silhouettes, key=silhouettes.get)
    st.caption(
        f"Highest silhouette score: k = {best_k} ({silhouettes[best_k]:.3f}). "
        "Silhouette ranges from -1 to 1; values close to 0 mean the clusters overlap. "
        "The elbow curve has no sharp bend, which also points to weak cluster structure."
    )

    k_chosen = st.slider("Choose k to compare clusters with the real labels", min_value=2, max_value=6, value=2)
    km_model = KMeans(n_clusters=k_chosen, n_init=10, random_state=42).fit(scaled_features(df))
    table, purity, ari = cluster_agreement(km_model.labels_, y_all)

    agree_col1, agree_col2 = st.columns([1, 1])
    with agree_col1:
        st.markdown(f"**Cluster vs actual label (k = {k_chosen})**")
        st.table(table)
    with agree_col2:
        st.metric("Silhouette score", f"{silhouettes[k_chosen]:.3f}")
        st.metric("Cluster purity", f"{purity * 100:.1f}%", help="Share of samples whose cluster's majority label equals their own label.")
        st.metric("Adjusted Rand Index", f"{ari:.3f}", help="0 = clusters unrelated to labels, 1 = perfect match.")
    st.caption(
        f"Purity ({purity * 100:.1f}%) should be compared with the {majority_share * 100:.1f}% majority-class baseline, "
        f"and the Adjusted Rand Index ({ari:.3f}) is close to 0 when clusters ignore the labels."
    )

    # ---- 3. Hierarchical clustering ------------------------------------------------
    st.markdown("---")
    st.markdown("### 3. Hierarchical Clustering (Dendrogram on a Random Sample)")
    sample_size = st.slider("Sample size (a full dendrogram of 3,276 rows would be unreadable)", 30, 150, 60, step=10)
    sample_idx = np.random.RandomState(42).choice(len(df), size=sample_size, replace=False)
    X_sample = scaled_features(df)[sample_idx]
    y_sample = y_all[sample_idx]

    Z = linkage(X_sample, method="ward")
    leaf_names = ["P" if v == 1 else "N" for v in y_sample]
    fig_dendro, ax_dendro = plt.subplots(figsize=(12, 4.5))
    dendrogram(Z, labels=leaf_names, leaf_font_size=8, color_threshold=0, above_threshold_color="#475569", ax=ax_dendro)
    for tick in ax_dendro.get_xmajorticklabels():
        tick.set_color("#10b981" if tick.get_text() == "P" else "#ef4444")
    ax_dendro.set_title(f"Ward Linkage Dendrogram ({sample_size} samples; P = Potable green, N = Non-Potable red)", fontweight="bold")
    ax_dendro.set_ylabel("Distance")
    sns.despine()
    plt.tight_layout()
    st.pyplot(fig_dendro)
    plt.close(fig_dendro)

    h_clusters = fcluster(Z, t=2, criterion="maxclust")
    h_table, h_purity, h_ari = cluster_agreement(h_clusters, y_sample)
    h_col1, h_col2 = st.columns([1, 1])
    with h_col1:
        st.markdown("**Cutting the tree into 2 clusters vs actual labels**")
        st.table(h_table)
    with h_col2:
        h_majority = max(y_sample.mean(), 1 - y_sample.mean())
        st.metric("Cluster purity", f"{h_purity * 100:.1f}%", help=f"Majority-class share in this sample: {h_majority * 100:.1f}%")
        st.metric("Adjusted Rand Index", f"{h_ari:.3f}")
    st.caption(
        f"Compare the purity ({h_purity * 100:.1f}%) with the {h_majority * 100:.1f}% majority share of this sample: "
        "if they are about equal, the hierarchical clusters tell us nothing about potability. "
        "Change the sample size to see how unstable this small-sample result is."
    )

    # ---- 4. Takeaway -----------------------------------------------------------------
    st.markdown("---")
    st.success(
        "**Takeaway for the viva:** PCA, K-Means and hierarchical clustering show no natural split between potable and "
        "non-potable samples in this dataset. This is consistent with the modest scores of the supervised models and is "
        "a reason to judge them with F1 and ROC-AUC instead of trusting accuracy alone."
    )


# -----------------------------------------------------------------------------
# TAB 4: Academic Report & Viva Guide
# -----------------------------------------------------------------------------
elif navigation == "📖 Academic Report & Viva Guide":
    st.markdown('<div class="main-title">📖 Academic Report & Viva Guide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Comprehensive documentation structured for college project defense and presentations.</div>',
        unsafe_allow_html=True,
    )

    doc_tab1, doc_tab2, doc_tab3 = st.tabs([
        "📑 Seven Academic Requirements",
        "🎤 10-Minute Presentation Script",
        "🎯 Viva Q&A Preparation",
    ])

    with doc_tab1:
        st.markdown("### The 7 Core Academic Phases")

        st.markdown("""
        #### 1. Problem Definition
        - **Problem Statement:** "Water quality depends on several physical and chemical parameters. This project uses machine learning to analyze historical water-quality data and classify water samples as potable or non-potable according to their recorded dataset labels."
        - **Objectives:** Build an end-to-end, reproducible classification system to assist environmental scientists and college students in analyzing water safety indicators using machine learning.
        - **Scope & Limitations:** Model predictions are statistical approximations derived from historical dataset labels and cannot replace accredited physical laboratory chemical testing.

        #### 2. Dataset Collection & Schema Verification
        - **Source:** Public Kaggle Water Potability Dataset (`adityakadiwal/water-potability`).
        - **Records:** 3,276 water samples.
        - **Target:** `Potability` (1 = Potable, 0 = Non-Potable).
        - **Features (9 verified):** pH, Hardness, Solids (TDS), Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity.
        - **Missing Values:** Sulfate (781), pH (491), Trihalomethanes (162).

        #### 3. Data Preprocessing & Leakage Prevention
        - **Duplicate Checking:** 0 duplicate rows found in dataset.
        - **Imputation:** Median imputation (`SimpleImputer(strategy='median')`) used to handle missing numerical features robustly without being skewed by outliers.
        - **Feature Scaling:** `StandardScaler` standardizes features to zero mean and unit variance.
        - **Train/Test Split:** 80% training (2,620 samples) and 20% testing (656 samples) using **Stratified Splitting** to preserve class balance.
        - **Data Leakage Prevention:** The preprocessing pipeline is fitted strictly on the training set (`X_train`) and only used to transform the test set (`X_test`).
        """)

        # Sections 4-7 are built from model_results.json so the numbers always match the latest training run
        _orig_lr = results["models"]["logistic_regression"]
        _orig_rf = results["models"]["random_forest_optimized"]
        _imp = results["improved_models"]
        _sel = _imp[results["selected_model"]["key"]]
        _sel_params = ", ".join(f"{k}={v}" for k, v in _sel["best_parameters"].items())
        st.markdown(f"""
        #### 4. Model Development
        - **Original models:** Logistic Regression (linear baseline) and Random Forest (non-linear ensemble).
        - **Syllabus bake-off:** 7 classifiers on the same split: Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM (RBF), Random Forest and AdaBoost.
        - **Pipeline Integration:** Every model sits in a Scikit-learn Pipeline (median imputation, scaling, classifier).

        #### 5. Performance Evaluation
        - Evaluated on identical held-out test data ({results["dataset_info"]["test_size"]} samples) with Accuracy, Precision, Recall, F1-Score, ROC-AUC and Confusion Matrix.
        - **Original result:** Logistic Regression reached {_orig_lr["accuracy"]*100:.2f}% accuracy only by predicting class 0 for every sample (F1 = {_orig_lr["f1_score"]:.2f}); the tuned Random Forest reached {_orig_rf["accuracy"]*100:.2f}% accuracy but found only {_orig_rf["recall"]*100:.0f}% of potable samples (F1 = {_orig_rf["f1_score"]:.4f}).
        - **Key point:** the majority-class baseline already scores 61% accuracy, so accuracy alone is a weak measure here.

        #### 6. Model Optimization
        - **Class imbalance:** `class_weight='balanced'` for the models that support it.
        - **Tuning:** `GridSearchCV` with `scoring='f1'` and 5-fold `StratifiedKFold`, run on training data only.
        - **Selected model:** {results["selected_model"]["name"]}, chosen by the highest cross-validation F1 ({_sel["cv_f1"]:.4f}) with parameters `{_sel_params}`. Test F1 = {_sel["f1_score"]:.4f}, recall = {_sel["recall"]:.4f}, ROC-AUC = {_sel["roc_auc"]:.4f}.

        #### 7. Result Analysis & Conclusion
        - Balancing the classes raised potable recall (Random Forest: {_orig_rf["recall"]:.2f} to {_imp["random_forest"]["recall"]:.2f}) at the cost of accuracy ({_orig_rf["accuracy"]*100:.1f}% to {_imp["random_forest"]["accuracy"]*100:.1f}%).
        - The best models have ROC-AUC of only about 0.64-0.66, and PCA/clustering show no natural split between the classes, so the features carry limited information about the label. The model is a screening aid, not a safety test.
        """)

    with doc_tab2:
        st.markdown("### ⏱️ 10-Minute Presentation Slide Breakdown")
        st.markdown("""
        Use this exact timeline when presenting your project to college professors:

        | Time | Slide / Topic | What to Say |
        | :--- | :--- | :--- |
        | **0:00 - 1:30** | **Slide 1: Introduction & Problem Statement** | "Good morning professors. Today I am presenting AquaGuard AI, a machine learning system designed to classify water samples as potable or non-potable based on 9 physicochemical parameters like pH, sulfates, and chloramines." |
        | **1:30 - 3:00** | **Slide 2: Dataset & Preprocessing** | "We used the real-world Kaggle Water Potability dataset of 3,276 records. Missing values in Sulfate and pH were handled with median imputation. We split the data 80/20 with stratified sampling and strictly prevented data leakage by fitting transformers only on training data." |
        | **3:00 - 5:00** | **Slide 3: Algorithms & Class Imbalance** | "We compared seven classifiers from our syllabus on the same split. Our original Logistic Regression scored 61% accuracy only by predicting non-potable every time, which is the majority-class baseline. So we used F1-score and ROC-AUC, not just accuracy." |
        | **5:00 - 6:30** | **Slide 4: Optimization & Unsupervised Analysis** | "We used class weights and GridSearchCV with stratified 5-fold cross-validation scored on F1. The best cross-validation F1 came from the SVM. PCA and K-Means show potable and non-potable samples overlap heavily, which explains why scores stay modest." |
        | **6:30 - 8:30** | **Slide 5: Live System Demonstration** | Open the **Water Quality Predictor** tab. Click 'Load Known Potable Sample' and click Predict. Show the probability output and model badge. Then load 'Non-Potable Sample' to show contrast. Point out the academic disclaimer. |
        | **8:30 - 10:00** | **Slide 6: Conclusion & Viva Wrap-up** | "AquaGuard AI demonstrates how machine learning pipelines can automate preliminary water analysis while maintaining strict academic and scientific ethics. Thank you, I am now open for questions." |
        """)

    with doc_tab3:
        st.markdown("### 🎯 Top College Viva Questions & Answers")
        st.markdown("""
        **Q1: Why did the original Logistic Regression achieve 60.98% accuracy but 0.00 F1-score?**
        > *Answer:* The dataset has 61% non-potable and 39% potable samples. The unweighted model predicted class 0 for every sample, which gives 60.98% accuracy by default but zero recall and zero F1. Accuracy alone is therefore misleading on imbalanced data.

        **Q1b: What did you change to handle the imbalance?**
        > *Answer:* We used `class_weight='balanced'` where scikit-learn supports it, tuned with GridSearchCV on F1 using stratified 5-fold cross-validation, and reported ROC-AUC as well. Recall for potable water went up, and accuracy went down; that trade-off is shown in the Model Performance tab.

        **Q1c: Why did you pick the SVM instead of the model with the best test score?**
        > *Answer:* The model was chosen by cross-validation F1 on the training data only. Choosing by test score would leak the test set into model selection. The top models are close, so the choice is not a large difference.

        **Q1d: What does the PCA and clustering analysis tell you?**
        > *Answer:* The first two principal components explain only about a quarter of the variance, the two classes overlap, K-Means silhouette is about 0.08 and the Adjusted Rand Index is close to 0. So the features do not form natural potable/non-potable groups, which fits the modest supervised scores.

        **Q2: How did you ensure there was no data leakage in your preprocessing?**
        > *Answer:* We used Scikit-Learn Pipelines where `SimpleImputer` and `StandardScaler` were fitted exclusively on the 80% training set (`X_train`). The held-out 20% test set was only transformed using the training parameters, ensuring test statistics never leaked into model training.

        **Q3: Why did you choose Median Imputation instead of Mean Imputation?**
        > *Answer:* Environmental chemical features like Sulfate and Solids contain extreme outliers. The median is a non-parametric statistic resistant to extreme values, whereas the mean would be distorted by outliers.

        **Q4: Why was Stratified Splitting necessary?**
        > *Answer:* Because the dataset exhibits a 61:39 class distribution. Stratified splitting guarantees that both the 80% training set and 20% test set maintain the exact same proportion of potable and non-potable records.

        **Q5: Can this system replace municipal laboratory water testing?**
        > *Answer:* No. As stated in our project disclaimer, this is an educational decision-support model based on historical patterns. Biological pathogen assays and regulatory standards require accredited wet-chemistry lab analysis.
        """)
