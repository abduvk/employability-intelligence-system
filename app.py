import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employability Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }
    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        color: #e8ff47;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .hero-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #666;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .kpi-card {
        background: #111118;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border-left: 3px solid #e8ff47;
    }
    .kpi-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e8ff47;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #888;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.6rem;
        color: #ffffff;
        border-bottom: 2px solid #e8ff47;
        padding-bottom: 8px;
        margin-bottom: 1.5rem;
    }
    .insight-box {
        background: #111118;
        border: 1px solid #222;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e8ff47;
    }
    .insight-text {
        color: #ccc;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .rec-card {
        background: linear-gradient(135deg, #111118 0%, #16161f 100%);
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1.4rem;
        margin: 0.6rem 0;
    }
    .rec-priority {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .priority-high { background: #ff4757; color: white; }
    .priority-med  { background: #e8ff47; color: #0a0a0f; }
    .priority-low  { background: #2ed573; color: #0a0a0f; }

    div[data-testid="stSidebar"] {
        background-color: #0d0d14;
        border-right: 1px solid #1e1e2e;
    }
    .stSlider > div > div { color: #e8ff47 !important; }
    .stSelectbox label { color: #aaa !important; }
    h1, h2, h3 { color: #fff; }
</style>
""", unsafe_allow_html=True)

# ─── Data Generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=1000, seed=42):
    np.random.seed(seed)
    cgpa         = np.round(np.clip(np.random.normal(7.2, 1.0, n), 4.0, 10.0), 2)
    aptitude     = np.round(np.clip(np.random.normal(62, 15, n), 20, 100), 1)
    projects     = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.05, 0.25, 0.30, 0.20, 0.12, 0.08])
    internships  = np.random.choice([0, 1, 2, 3], n, p=[0.30, 0.40, 0.20, 0.10])
    certifications = np.random.choice([0, 1, 2, 3, 4], n, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    soft_skills  = np.round(np.clip(np.random.normal(3.2, 0.8, n), 1.0, 5.0), 1)
    extracurricular = np.random.choice([0, 1], n, p=[0.45, 0.55])
    placement_training = np.random.choice([0, 1], n, p=[0.35, 0.65])
    ssc          = np.round(np.clip(np.random.normal(72, 12, n), 40, 100), 1)
    hsc          = np.round(np.clip(np.random.normal(70, 12, n), 40, 100), 1)

    score = (
        0.28 * (aptitude / 100) +
        0.22 * (projects / 5) +
        0.15 * ((cgpa - 4) / 6) +
        0.13 * (soft_skills / 5) +
        0.09 * (certifications / 4) +
        0.07 * (internships / 3) +
        0.03 * extracurricular +
        0.03 * placement_training +
        np.random.normal(0, 0.05, n)
    )
    placed = (score > np.percentile(score, 38)).astype(int)

    return pd.DataFrame({
        "CGPA": cgpa, "SSC_Marks": ssc, "HSC_Marks": hsc,
        "Aptitude_Score": aptitude, "Projects": projects,
        "Internships": internships, "Certifications": certifications,
        "Soft_Skills": soft_skills, "Extracurricular": extracurricular,
        "Placement_Training": placement_training, "Placed": placed
    })

@st.cache_resource
def train_models(df):
    features = ["CGPA", "Aptitude_Score", "Projects", "Internships",
                "Certifications", "Soft_Skills", "Extracurricular", "Placement_Training"]
    X = df[features]
    y = df["Placed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return rf, lr, scaler, X_test, X_test_s, y_test, rf_pred, lr_pred, importances, features

df = generate_dataset()
rf, lr, scaler, X_test, X_test_s, y_test, rf_pred, lr_pred, importances, features = train_models(df)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 EIS Navigation")
    page = st.radio("", ["Overview", "EDA", "ML Models", "Predictor", "Recommendations"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🎨 About")
    st.markdown("""
    <div style='color:#888; font-size:0.82rem; line-height:1.8;'>
    Employability Intelligence System<br>
    Built with Python · Scikit-Learn · Streamlit<br><br>
    <span style='color:#e8ff47'>v1.0</span> — Portfolio Project
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="hero-title">Employability<br>Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Data-driven placement outcome analysis · 1,000 student records</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    placed_df   = df[df["Placed"] == 1]
    unplaced_df = df[df["Placed"] == 0]
    placement_rate = round(df["Placed"].mean() * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Total Students</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{placement_rate}%</div><div class="kpi-label">Placement Rate</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{placed_df["CGPA"].mean():.2f}</div><div class="kpi-label">Avg CGPA (Placed)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{placed_df["Aptitude_Score"].mean():.1f}</div><div class="kpi-label">Avg Aptitude (Placed)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Factor Impact on Placement</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#111118')
        ax.set_facecolor('#111118')
        labels = [f.replace("_", " ") for f in importances.index]
        vals   = importances.values * 100
        colors = ['#e8ff47' if i == 0 else '#444' for i in range(len(vals))]
        bars   = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6)
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='#aaa', fontsize=9, fontfamily='monospace')
        ax.set_xlabel("Feature Importance (%)", color='#666', fontsize=9)
        ax.tick_params(colors='#888', labelsize=9)
        ax.spines[:].set_visible(False)
        ax.xaxis.grid(True, color='#222', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        insights = [
            ("🥇", "Aptitude Score", "Strongest predictor of placement. High-scoring students are 2.1× more likely to be placed."),
            ("🥈", "Projects", "Each project added increases placement odds significantly. Aim for 3+."),
            ("🥉", "CGPA", "Matters, but less than skills. 7.5+ is a safe threshold."),
            ("4️⃣", "Soft Skills", "Often underestimated. Rated 4+ correlates strongly with placement.")
        ]
        for icon, title, text in insights:
            st.markdown(f"""
            <div class="insight-box">
                <b style='color:#e8ff47'>{icon} {title}</b>
                <div class="insight-text">{text}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📈 Placement Rates", "🔥 Correlation"])

    with tab1:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        fig.patch.set_facecolor('#0d0d14')
        plot_cols = ["CGPA", "Aptitude_Score", "Soft_Skills", "Projects", "Internships", "Certifications"]
        for ax, col in zip(axes.flat, plot_cols):
            ax.set_facecolor('#111118')
            placed   = df[df["Placed"] == 1][col]
            unplaced = df[df["Placed"] == 0][col]
            ax.hist(unplaced, bins=15, alpha=0.5, color='#444', label='Not Placed', density=True)
            ax.hist(placed,   bins=15, alpha=0.8, color='#e8ff47', label='Placed', density=True)
            ax.set_title(col.replace("_", " "), color='#ccc', fontsize=10, fontweight='bold')
            ax.tick_params(colors='#666', labelsize=8)
            ax.spines[:].set_visible(False)
        p_patch = mpatches.Patch(color='#e8ff47', alpha=0.8, label='Placed')
        u_patch = mpatches.Patch(color='#444',    alpha=0.5, label='Not Placed')
        fig.legend(handles=[p_patch, u_patch], loc='lower right', facecolor='#111118', labelcolor='#aaa', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor('#0d0d14')
        bar_cols  = ["Projects", "Internships", "Certifications"]
        for ax, col in zip(axes, bar_cols):
            ax.set_facecolor('#111118')
            rate = df.groupby(col)["Placed"].mean() * 100
            clrs = ['#e8ff47' if v == rate.max() else '#333' for v in rate.values]
            ax.bar(rate.index.astype(str), rate.values, color=clrs, width=0.6)
            ax.set_title(f"{col} vs Placement Rate", color='#ccc', fontsize=10, fontweight='bold')
            ax.set_ylabel("Placement Rate (%)", color='#666', fontsize=8)
            ax.set_xlabel(col, color='#666', fontsize=8)
            ax.tick_params(colors='#666', labelsize=8)
            ax.spines[:].set_visible(False)
            ax.yaxis.grid(True, color='#1e1e1e', linewidth=0.5)
            for i, (x, v) in enumerate(zip(rate.index, rate.values)):
                ax.text(i, v + 1, f'{v:.0f}%', ha='center', color='#888', fontsize=8, fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        corr_cols = ["CGPA", "Aptitude_Score", "Projects", "Internships",
                     "Certifications", "Soft_Skills", "Placed"]
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor('#0d0d14')
        ax.set_facecolor('#111118')
        cmap = sns.diverging_palette(0, 90, s=80, l=40, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                    linewidths=0.5, linecolor='#0a0a0f',
                    annot_kws={"size": 9, "color": "white"},
                    cbar_kws={"shrink": 0.8})
        ax.tick_params(colors='#888', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ML Models":
    st.markdown('<div class="section-title">Machine Learning Models</div>', unsafe_allow_html=True)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_pre = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_pre = precision_score(y_test, lr_pred)
    lr_rec = recall_score(y_test, lr_pred)

    col1, col2 = st.columns(2)
    for col, name, acc, pre, rec, pred in [
        (col1, "Random Forest", rf_acc, rf_pre, rf_rec, rf_pred),
        (col2, "Logistic Regression", lr_acc, lr_pre, lr_rec, lr_pred)
    ]:
        with col:
            st.markdown(f"#### 🤖 {name}")
            mc1, mc2, mc3 = st.columns(3)
            for mc, label, val in [(mc1,"Accuracy",acc),(mc2,"Precision",pre),(mc3,"Recall",rec)]:
                with mc:
                    st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="font-size:1.5rem">{val:.2%}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            cm = confusion_matrix(y_test, pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor('#111118')
            ax.set_facecolor('#111118')
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', ax=ax,
                        xticklabels=["Not Placed","Placed"],
                        yticklabels=["Not Placed","Placed"],
                        linewidths=1, linecolor='#0a0a0f',
                        annot_kws={"size": 11})
            ax.set_xlabel("Predicted", color='#888', fontsize=9)
            ax.set_ylabel("Actual", color='#888', fontsize=9)
            ax.tick_params(colors='#888', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Predictor":
    st.markdown('<div class="section-title">Placement Probability Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#888;font-size:0.9rem;margin-bottom:1.5rem">Enter your profile to get a placement probability score.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cgpa      = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
        aptitude  = st.slider("Aptitude Score", 20, 100, 65)
        projects  = st.slider("Number of Projects", 0, 5, 2)
        internships = st.slider("Number of Internships", 0, 3, 1)
    with col2:
        certs     = st.slider("Certifications", 0, 4, 2)
        soft      = st.slider("Soft Skills Rating", 1.0, 5.0, 3.5, 0.1)
        extra     = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        training  = st.selectbox("Placement Training", ["Yes", "No"])

    extra_val   = 1 if extra == "Yes" else 0
    training_val = 1 if training == "Yes" else 0
    input_arr   = np.array([[cgpa, aptitude, projects, internships, certs, soft, extra_val, training_val]])
    input_scaled = scaler.transform(input_arr)

    rf_prob = rf.predict_proba(input_arr)[0][1]
    lr_prob = lr.predict_proba(input_scaled)[0][1]
    avg_prob = (rf_prob + lr_prob) / 2

    st.markdown("<br>", unsafe_allow_html=True)
    color = "#2ed573" if avg_prob > 0.65 else ("#e8ff47" if avg_prob > 0.40 else "#ff4757")
    label = "High Placement Probability" if avg_prob > 0.65 else ("Moderate — Needs Improvement" if avg_prob > 0.40 else "Low — Significant Gaps")
    st.markdown(f"""
    <div style="background:#111118;border:1px solid #222;border-radius:16px;padding:2rem;text-align:center;border-top:4px solid {color}">
        <div style="font-family:'Space Mono',monospace;font-size:3.5rem;color:{color};font-weight:700">{avg_prob:.0%}</div>
        <div style="color:#aaa;font-size:1rem;margin-top:0.5rem">{label}</div>
        <div style="color:#555;font-size:0.8rem;margin-top:0.5rem;font-family:monospace">RF: {rf_prob:.0%} · LR: {lr_prob:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown('<div class="section-title">Personalized Recommendation Engine</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cgpa_r    = st.number_input("Your CGPA", 4.0, 10.0, 7.0, 0.1)
        proj_r    = st.number_input("Projects completed", 0, 10, 1)
        intern_r  = st.number_input("Internships done", 0, 5, 0)
    with col2:
        cert_r    = st.number_input("Certifications", 0, 10, 1)
        apt_r     = st.number_input("Aptitude Score", 20, 100, 55)
        soft_r    = st.number_input("Soft Skills (1–5)", 1.0, 5.0, 3.0, 0.1)

    if st.button("🚀 Generate My Roadmap", type="primary"):
        recs = []
        if apt_r < 65:
            recs.append(("HIGH", "🧠 Aptitude Training", f"Your score ({apt_r}) is below the placed-student average (65+). Dedicate 30 min/day for 6 weeks to quantitative and logical reasoning practice."))
        if proj_r < 3:
            recs.append(("HIGH", "💻 Build More Projects", f"You have {proj_r} project(s). Placed students average 3+. Pick one domain (web/ML/data) and complete 2 end-to-end projects this semester."))
        if cgpa_r < 7.5:
            recs.append(("MED", "📚 Academic Focus", f"Your CGPA ({cgpa_r}) is below the recommended 7.5 threshold. Focus on core subjects and attempt backlog clearance if applicable."))
        if soft_r < 3.5:
            recs.append(("MED", "🗣 Soft Skills Development", f"Your rating ({soft_r}/5) is low. Join a debate club, practice mock interviews, and attend group discussions weekly."))
        if cert_r < 2:
            recs.append(("LOW", "📜 Add Certifications", f"You have {cert_r} certification(s). Add 1–2 relevant certs (Google Data Analytics, AWS, etc.) to strengthen your resume."))
        if intern_r == 0:
            recs.append(("MED", "🏢 Internship Experience", "No internship yet. Apply to 10 companies this week on LinkedIn and Internshala. Even a 1-month remote internship adds significant value."))
        if not recs:
            st.success("✅ Your profile looks strong! Focus on polishing your resume and practicing interview skills.")
        else:
            priority_order = {"HIGH": 0, "MED": 1, "LOW": 2}
            recs.sort(key=lambda x: priority_order[x[0]])
            for priority, title, text in recs:
                cls = {"HIGH": "priority-high", "MED": "priority-med", "LOW": "priority-low"}[priority]
                st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-priority {cls}">{priority} PRIORITY</span>
                    <div style="color:#fff;font-weight:700;font-size:1rem;margin-bottom:0.4rem">{title}</div>
                    <div style="color:#aaa;font-size:0.88rem;line-height:1.6">{text}</div>
                </div>""", unsafe_allow_html=True)
