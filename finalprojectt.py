import streamlit as st
st.write("APP STARTED")

import matplotlib.pyplot as plt
st.write("MATPLOTLIB OK")

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, shapiro, linregress

# =====================
# Page Config (Light mode)
# =====================
st.set_page_config(page_title="Survey Data Analysis", layout="wide")

# Optional: force light-ish look via CSS (tidy, not mandatory)
st.markdown(
    """
    <style>
      .stApp { background-color: #ffffff; }
      [data-testid="stSidebar"] { background-color: #f6f7fb; }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================
# Language Dictionary
# =====================
LANG = {
    "id": {
        "title": "📊 Aplikasi Analisis Data Survei",
        "desc": "Analisis deskriptif dan asosiasi (korelasi). Uji normalitas digunakan untuk memilih Pearson atau Spearman.",
        "upload": "Upload file dataset (CSV atau Excel .xlsx)",
        "preview": "Pratinjau Data",
        "desc_stat": "Analisis Deskriptif (Numerik)",
        "select_numeric": "Pilih kolom numerik (untuk statistik deskriptif)",
        "select_x": "Pilih Variabel X",
        "select_y": "Pilih Variabel Y",
        "analyze": "Jalankan Analisis Asosiasi",
        "result": "📌 Hasil Analisis",
        "analysis_desc": "📝 Deskripsi Analisis",
        "method": "Metode Korelasi",
        "corr": "Koefisien Korelasi",
        "pval": "p-value",
        "normality": "Uji Normalitas (Shapiro-Wilk)",
        "px": "p-value X",
        "py": "p-value Y",
        "info": "Silakan upload file untuk memulai.",
        "warn_no_numeric": "Tidak ada kolom numerik terdeteksi. Coba ubah kolom ke angka atau gunakan fitur konversi.",
        "convert_hint": "Coba konversi kolom bertipe teks ke numerik (opsional).",
        "convert_cols": "Pilih kolom yang ingin dicoba dikonversi ke numerik",
        "apply_convert": "Terapkan konversi",
        "error_pairs": "Data valid tidak cukup untuk analisis (minimal 3 baris setelah hapus missing).",
        "positive": "positif",
        "negative": "negatif",
        "very_weak": "sangat lemah / hampir tidak ada",
        "weak": "lemah",
        "moderate": "sedang",
        "strong": "kuat",
        "scatter": "Scatter Plot + Garis Regresi",
        "convert_success": "Konversi diterapkan.",
    },
    "en": {
        "title": "📊 Survey Data Analysis App",
        "desc": "Descriptive and association analysis (correlation). Normality testing helps choose Pearson or Spearman.",
        "upload": "Upload dataset file (CSV or Excel .xlsx)",
        "preview": "Data Preview",
        "desc_stat": "Descriptive Analysis (Numeric)",
        "select_numeric": "Select numeric columns (for descriptive statistics)",
        "select_x": "Select Variable X",
        "select_y": "Select Variable Y",
        "analyze": "Run Association Analysis",
        "result": "📌 Analysis Result",
        "analysis_desc": "📝 Analysis Description",
        "method": "Correlation Method",
        "corr": "Correlation Coefficient",
        "pval": "p-value",
        "normality": "Normality Test (Shapiro-Wilk)",
        "px": "p-value X",
        "py": "p-value Y",
        "info": "Please upload a file to start.",
        "warn_no_numeric": "No numeric columns detected. Try converting text columns to numeric.",
        "convert_hint": "Try converting text columns to numeric (optional).",
        "convert_cols": "Select columns to attempt numeric conversion",
        "apply_convert": "Apply conversion",
        "error_pairs": "Not enough valid data (need at least 3 rows after dropping missing).",
        "positive": "positive",
        "negative": "negative",
        "very_weak": "very weak / almost none",
        "weak": "weak",
        "moderate": "moderate",
        "strong": "strong",
        "scatter": "Scatter Plot + Regression Line",
        "convert_success": "Conversion applied.",
    }
}

# =====================
# Language Selector
# =====================
st.sidebar.title("🌐 Language / Bahasa")
lang = st.sidebar.radio(
    "Select Language",
    ["id", "en"],
    format_func=lambda x: "🇮🇩 Bahasa Indonesia" if x == "id" else "🇬🇧 English"
)
T = LANG[lang]

# =====================
# Title
# =====================
st.title(T["title"])
st.write(T["desc"])

# =====================
# Upload File
# =====================
uploaded_file = st.file_uploader(T["upload"], type=["xlsx", "csv"])

def read_data(file):
    """Safely read CSV or Excel file"""
    try:
        name = file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

if uploaded_file:
    df = read_data(uploaded_file)
    
    if df is None:
        st.stop()

    st.subheader(T["preview"])
    st.dataframe(df.head())

    # --- Optional conversion tool (helps when numeric columns are read as text) ---
    st.caption(T["convert_hint"])
    convert_candidates = [c for c in df.columns if df[c].dtype == "object"]
    if convert_candidates:
        cols_to_convert = st.multiselect(T["convert_cols"], options=convert_candidates)
        if st.button(T["apply_convert"]) and cols_to_convert:
            for c in cols_to_convert:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            st.success(T["convert_success"])
            st.dataframe(df.head())

    # Detect numeric columns after conversion attempt
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning(T["warn_no_numeric"])
        st.stop()

    # =====================
    # Descriptive Analysis
    # =====================
    st.subheader(T["desc_stat"])

    default_count = min(10, len(numeric_cols))
    desc_cols = st.multiselect(
        T["select_numeric"],
        options=numeric_cols,
        default=numeric_cols[:default_count]
    )

    if desc_cols:
        desc_stats = df[desc_cols].describe().T
        st.dataframe(desc_stats.round(4))

    # =====================
    # Association Analysis
    # =====================
    st.subheader(T["select_x"])
    
    # Ensure we have at least 2 numeric columns for correlation
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis." if lang == "en" else "Membutuhkan minimal 2 kolom numerik untuk analisis korelasi.")
        st.stop()
    
    col_x = st.selectbox(T["select_x"], numeric_cols, index=0, key="select_x")
    col_y = st.selectbox(T["select_y"], numeric_cols, index=1, key="select_y")

    if st.button(T["analyze"]):
        # Prepare data
        data = df[[col_x, col_y]].copy()
        data[col_x] = pd.to_numeric(data[col_x], errors="coerce")
        data[col_y] = pd.to_numeric(data[col_y], errors="coerce")
        data = data.dropna()

        if len(data) < 3:
            st.error(T["error_pairs"])
            st.stop()

        # -------- Normality test (safe) --------
        def safe_shapiro(x):
            """Safely perform Shapiro-Wilk test"""
            try:
                x = np.asarray(x).flatten()
                if len(x) < 3:
                    return np.nan
                # Check for constant values
                if np.std(x) == 0 or np.all(x == x[0]):
                    return np.nan
                # Check for valid range
                if len(x) > 5000:
                    # Shapiro-Wilk can be unreliable for very large samples
                    return np.nan
                stat, p = shapiro(x)
                return float(p)
            except Exception:
                return np.nan

        p_x = safe_shapiro(data[col_x])
        p_y = safe_shapiro(data[col_y])

        st.subheader(T["normality"])
        px_text = "N/A" if np.isnan(p_x) else f"{p_x:.4f}"
        py_text = "N/A" if np.isnan(p_y) else f"{p_y:.4f}"
        st.write(f"{T['px']}: {px_text}")
        st.write(f"{T['py']}: {py_text}")

        # Rule: If both p-values available and > 0.05 -> Pearson; otherwise Spearman
        use_pearson = (not np.isnan(p_x)) and (not np.isnan(p_y)) and (p_x > 0.05) and (p_y > 0.05)

        try:
            if use_pearson:
                method = "Pearson"
                corr, pval = pearsonr(data[col_x], data[col_y])
            else:
                method = "Spearman"
                corr, pval = spearmanr(data[col_x], data[col_y])
        except Exception as e:
            st.error(f"Error calculating correlation: {str(e)}")
            st.stop()

        direction = T["positive"] if corr > 0 else T["negative"]

        # Strength interpretation
        abs_r = abs(corr)
        if abs_r < 0.3:
            strength = T["very_weak"]
        elif abs_r < 0.5:
            strength = T["weak"]
        elif abs_r < 0.7:
            strength = T["moderate"]
        else:
            strength = T["strong"]

        # =====================
        # Output
        # =====================
        st.subheader(T["result"])
        c1, c2, c3 = st.columns(3)
        c1.metric(T["method"], method)
        c2.metric(T["corr"], f"{corr:.4f}")
        c3.metric(T["pval"], f"{pval:.4f}")

        # =====================
        # Visualization (stable regression line)
        # =====================
        st.subheader(T["scatter"])

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(data[col_x], data[col_y], alpha=0.7, edgecolors='k', linewidth=0.5)

        # Regression line safely with linregress
        try:
            x_vals = data[col_x].values
            y_vals = data[col_y].values
            
            # Check for sufficient variance
            if np.std(x_vals) > 0:
                lr = linregress(x_vals, y_vals)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = lr.intercept + lr.slope * x_line
                ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'y = {lr.slope:.3f}x + {lr.intercept:.3f}')
                ax.legend()
        except Exception:
            # If regression fails, continue without line
            pass

        ax.set_xlabel(col_x, fontsize=12)
        ax.set_ylabel(col_y, fontsize=12)
        ax.set_title(f"{col_x} vs {col_y}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # =====================
        # Description
        # =====================
        st.subheader(T["analysis_desc"])
        if lang == "id":
            st.markdown(
                f"""
                - Metode yang digunakan: **{method}**
                - Koefisien korelasi: **{corr:.3f}** (arah **{direction}**, kekuatan **{strength}**)
                - p-value: **{pval:.4f}**
                """
            )
        else:
            st.markdown(
                f"""
                - Method used: **{method}**
                - Correlation coefficient: **{corr:.3f}** (direction **{direction}**, strength **{strength}**)
                - p-value: **{pval:.4f}**
                """
            )

else:
    st.info(T["info"])
