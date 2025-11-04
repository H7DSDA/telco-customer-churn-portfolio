
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn – Portfolio Dashboard", layout="wide")

# ---------- Sidebar: Paths & Options ----------
st.sidebar.header("Paths & Files")

base_folder = st.sidebar.text_input(
    "Base folder",
    value="/content/drive/MyDrive/Portofolio/1_Telco Customer Churn",
    help="Folder utama proyek di Google Drive/Colab"
)

clean_csv = st.sidebar.text_input(
    "Clean CSV:",
    value=os.path.join(base_folder, "data_clean", "churn_clean.csv"),
)

model_path = st.sidebar.text_input(
    "Model file:",
    value=os.path.join(base_folder, "models", "xgb_smote.joblib"),
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption("Tips: pastikan path di atas sesuai struktur project-mu.")

# ---------- Safe loaders ----------
@st.cache_data(show_spinner=False)
def load_df(path):
    df = pd.read_csv(path)
    return df

@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)

def safe_load_df(p):
    try:
        return load_df(p), None
    except Exception as e:
        return None, e

def safe_load_model(p):
    try:
        return load_model(p), None
    except Exception as e:
        return None, e

# ---------- Main logic ----------
col_l, col_r = st.columns([1, 1])
with col_l:
    st.title("📊 Telco Customer Churn – Portfolio Dashboard")
with col_r:
    st.write("")
    st.write("")
    st.info("Built for portfolio by **Hans Christian** – Streamlit app")

df, err_df = safe_load_df(clean_csv)
model, err_m = safe_load_model(model_path)

tab_overview, tab_eval, tab_importance, tab_notes = st.tabs(
    ["Overview", "Model Evaluation", "Feature Importance (Permutation)", "Notes & Recommendations"]
)

with tab_overview:
    st.subheader("Dataset Overview")
    if err_df:
        st.error(f"Cannot read CSV: {err_df}")
    else:
        st.write("Shape:", df.shape)
        st.dataframe(df.head(20))
        st.markdown("**Numeric summary**")
        st.dataframe(df.describe(include="all").transpose())

with tab_eval:
    st.subheader("Model Evaluation")
    if err_m:
        st.error(f"Cannot load model: {err_m}")
    elif err_df:
        st.warning("Model loaded, but dataset not available.")
    else:
        # pick target column
        target_col = st.selectbox(
            "Target column (label)",
            options=[c for c in df.columns if df[c].nunique() <= 10] + list(df.columns),
            index= list(df.columns).index("Churn") if "Churn" in df.columns else 0
        )

        # split X/y
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y))>1 else None
        )

        # predict
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            # If pipeline returns no predict_proba (e.g., thresholded model)
            y_proba = None
        y_pred = model.predict(X_test)

        # metrics
        acc = accuracy_score(y_test, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        st.metric("Accuracy", f"{acc:.3f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{pr:.3f}")
        c2.metric("Recall", f"{rc:.3f}")
        c3.metric("F1-score", f"{f1:.3f}")

        st.markdown("**Classification report**")
        st.code(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        for (i,j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig, use_container_width=True)

        # ROC
        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            ax2.plot([0,1],[0,1], linestyle="--")
            ax2.set_xlabel("FPR")
            ax2.set_ylabel("TPR")
            ax2.set_title("ROC Curve")
            ax2.legend(loc="lower right")
            st.pyplot(fig2, use_container_width=True)
        else:
            st.warning("Model does not support predict_proba; ROC-AUC skipped.")

with tab_importance:
    st.subheader("Permutation Importance (fast & model-agnostic)")
    if any([err_df, err_m]):
        st.info("Load dataset & model first.")
    else:
        from sklearn.inspection import permutation_importance

        target_col = st.selectbox(
            "Target column for importance",
            options=[c for c in df.columns if c != "Churn"] + ["Churn"],
            index=list(df.columns).index("Churn") if "Churn" in df.columns else 0,
            key="perm_target"
        )
        X = df.drop(columns=[target_col]); y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y))>1 else None
        )
        st.caption("Running permutation importance on the test split...")
        try:
            scorer = None  # default uses model.score
            pi = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, scoring=scorer)
            importances = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)[:25]
            st.bar_chart(importances)
            st.dataframe(importances.rename("importance").to_frame())
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")

with tab_notes:
    st.subheader("Notes & Recommendations")
    st.markdown(\"\"\"
- **Tenure pendek** biasanya berisiko churn lebih tinggi → program onboarding & edukasi produk.
- **Month-to-Month contract** → tawarkan insentif upgrade ke 1–2 tahun.
- **Electronic check**/manual payment → arahkan ke auto-payment untuk menurunkan churn.
- Monitor segmentasi pengguna dengan **dashboard ini** dan tagging churn-prone customers.
\"\"\")

st.caption("© Portfolio app – Hans Christian")
