# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==============================
# PAGE SETUP
# ==============================
st.set_page_config(page_title="Ultrasonic AWY Detection", layout="wide")
st.title("🎯 Ultrasonic Sensor: Speed & Motion Dashboard")


# ==============================
# DATA LOADING
# ==============================
@st.cache_data
def load_data():
    return pd.read_excel("D:/IT/3rd Semester/ML/ultrasonic_project/master_dataset.xlsx")


try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ master_dataset.xlsx not found.")
    st.stop()


# Prepare ML Data
# Keep all columns for display, but separate X and y for ML
feature_cols = [c for c in df.columns if c not in ['filename', 'label', 'reg_mae', 'reg_rmse', 'awy_detected']]
X = df[feature_cols]
y = df['label'].values
filenames = df['filename'].values
class_names = sorted(df['label'].unique())

# ==============================
# UI TABS
# ==============================
# A professional layout uses tabs to prevent endless scrolling
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Extracted Features",
    "🏃 2. Speed Estimation Summary",
    "🧠 3. Model Evaluation",
    "🎯 4. Predictions & Importance"
])

# ---------------------------------------------------------
# TAB 1: EXTRACTED FEATURES (Your big console table)
# ---------------------------------------------------------
with tab1:
    st.subheader("Raw Extracted Features (Per-File)")

    # Display top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Files", len(df))
    col2.metric("Total Features", len(feature_cols))
    col3.metric("Overall Mean MAE", f"{df['reg_mae'].mean():.4f} m")
    col4.metric("Overall Mean RMSE", f"{df['reg_rmse'].mean():.4f} m")
    col5.metric("AWY Detected", f"{df['awy_detected'].sum()}/{len(df)}")

    # Show the interactive dataframe
    # Reorder columns slightly to match your console preference
    display_cols = ['filename', 'label', 'start_distance', 'end_distance', 'displacement',
                    'duration', 'regression_speed', 'awy_detected', 'reg_mae', 'reg_rmse',
                    'num_dropouts', 'dropout_ratio', 'mean_amplitude', 'std_amplitude',
                    'mean_echo_energy', 'std_echo_energy', 'signal_range', 'start_stability']
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 2: SPEED ESTIMATION SUMMARY
# ---------------------------------------------------------
with tab2:
    st.subheader("Speed Estimation - Regression Fit Quality")
    st.markdown("`d(t) = d0 + v*t  |  slope v = estimated speed`")

    # Group data by class just like your console output
    summary_data = []
    for cls in class_names:
        cls_df = df[df['label'] == cls]
        summary_data.append({
            "Class": cls,
            "Count": len(cls_df),
            "Min Speed (m/s)": cls_df['regression_speed'].min(),
            "Max Speed (m/s)": cls_df['regression_speed'].max(),
            "Mean Speed (m/s)": cls_df['regression_speed'].mean(),
            "Mean MAE (m)": cls_df['reg_mae'].mean(),
            "Mean RMSE (m)": cls_df['reg_rmse'].mean(),
            "AWY Detected": f"{cls_df['awy_detected'].sum()}/{len(cls_df)}"
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df.style.format({
        "Min Speed (m/s)": "{:.4f}", "Max Speed (m/s)": "{:.4f}",
        "Mean Speed (m/s)": "{:.4f}", "Mean MAE (m)": "{:.4f}", "Mean RMSE (m)": "{:.4f}"
    }), use_container_width=True, hide_index=True)

    # Overall summary across all files
    st.markdown("#### Overall Summary (All Files)")
    ov_col1, ov_col2, ov_col3 = st.columns(3)
    ov_col1.metric("Overall Mean MAE", f"{df['reg_mae'].mean():.4f} m")
    ov_col2.metric("Overall Mean RMSE", f"{df['reg_rmse'].mean():.4f} m")
    ov_col3.metric("AWY Detected", f"{df['awy_detected'].sum()}/{len(df)}")

# ---------------------------------------------------------
# TAB 3: MODEL EVALUATION (LOO-CV)
# ---------------------------------------------------------
with tab3:
    st.subheader("Model Comparison (Leave-One-Out Cross-Validation)")

    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=3),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=3),
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(kernel='rbf', random_state=42),
    }

    loo = LeaveOneOut()
    scaler = StandardScaler()
    all_preds = {}
    best_acc = 0
    best_model_name = ""

    # Train and evaluate models
    with st.spinner('Training models...'):
        for mname, model in models.items():
            yt, yp = [], []
            for tri, tei in loo.split(X):
                Xtr, Xte = X.iloc[tri], X.iloc[tei]
                ytr, yte = y[tri], y[tei]
                Xtr_s = scaler.fit_transform(Xtr)
                Xte_s = scaler.transform(Xte)
                model.fit(Xtr_s, ytr)
                yp.append(model.predict(Xte_s)[0])
                yt.append(yte[0])

            acc = accuracy_score(yt, yp)
            all_preds[mname] = (yt, yp, acc)
            if acc > best_acc:
                best_acc = acc
                best_model_name = mname

    st.success(f"🏆 Best Model: **{best_model_name}** with **{best_acc:.1%}** Accuracy")

    # Display Accuracies and Classification Reports
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("### Accuracy Scores")
        for mname in models:
            acc = all_preds[mname][2]
            n_correct = int(acc * len(y))
            st.metric(mname, f"{acc:.1%}", f"{n_correct}/{len(y)} correct", delta_color="off")

    with col_b:
        st.markdown(f"### Detailed Metrics ({best_model_name})")
        yt, yp, _ = all_preds[best_model_name]
        report_dict = classification_report(yt, yp, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    # Confusion Matrices Plot
    st.markdown("### Confusion Matrices")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, mname in enumerate(models):
        yt, yp, acc = all_preds[mname]
        cm = confusion_matrix(yt, yp, labels=class_names)
        ax = axes[idx // 2][idx % 2]
        ax.imshow(cm, cmap='Blues')
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{mname} ({acc:.1%})')
    plt.tight_layout()
    st.pyplot(fig)

    # Detailed metrics for ALL models
    st.markdown("### Detailed Metrics (All Models)")
    for mname in models:
        yt, yp, acc = all_preds[mname]
        with st.expander(f"📋 {mname} ({acc:.1%}) — Precision, Recall, F1"):
            report_dict_all = classification_report(yt, yp, target_names=class_names, output_dict=True)
            report_df_all = pd.DataFrame(report_dict_all).transpose()
            st.dataframe(report_df_all.style.format("{:.3f}"), use_container_width=True)
            cm = confusion_matrix(yt, yp, labels=class_names)
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_df.index.name = "Actual \\ Predicted"
            st.markdown("**Confusion Matrix (counts):**")
            st.dataframe(cm_df, use_container_width=False)

# ---------------------------------------------------------
# TAB 4: PREDICTIONS & FEATURE IMPORTANCE
# ---------------------------------------------------------
with tab4:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"Per-File Predictions ({best_model_name})")
        yt, yp, _ = all_preds[best_model_name]

        pred_data = []
        for i in range(len(filenames)):
            status = "✅ OK" if yt[i] == yp[i] else "❌ WRONG"
            pred_data.append({
                "#": i + 1,
                "Filename": filenames[i],
                "Actual": yt[i],
                "Predicted": yp[i],
                "Result": status
            })

        pred_df = pd.DataFrame(pred_data)
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # Score summary
        wrong = sum(a != b for a, b in zip(yt, yp))
        st.markdown(f"**Score: {len(filenames) - wrong}/{len(filenames)} correct, {wrong} wrong**")

    with col_right:
        st.subheader("Feature Importance (Random Forest)")
        # Calculate feature importance
        rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=3)
        rf.fit(scaler.fit_transform(X), y)
        imp = rf.feature_importances_

        # Create a dataframe for the chart
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": imp
        }).sort_values(by="Importance", ascending=True)  # Ascending for horizontal bar chart

        # Plot using Streamlit's native chart
        st.bar_chart(imp_df.set_index("Feature"), horizontal=True)

        # Ranked importance table
        st.markdown("#### Ranked Feature Importance")
        imp_ranked = imp_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
        imp_ranked.index += 1
        imp_ranked["Importance"] = imp_ranked["Importance"].map("{:.4f}".format)
        st.dataframe(imp_ranked, use_container_width=True)