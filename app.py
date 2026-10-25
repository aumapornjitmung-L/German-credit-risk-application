"""
German Credit Risk — Model, Explainability & Fairness
Streamlit app  |  รัน: streamlit run app.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

import credit_core as cc

warnings.filterwarnings("ignore")
st.set_page_config(page_title="German Credit Risk", page_icon="💳", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="กำลังดาวน์โหลดข้อมูลจาก OpenML (data_id=31) ...")
def load_openml():
    return cc.load_german_credit()


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype("category")
    return df


@st.cache_resource(show_spinner=False)
def train(_df, data_key, drop_sensitive, fast, c_fn, c_fp, test_size):
    bar = st.progress(0.0, text="เริ่มฝึกโมเดล ...")
    exp = cc.run_experiment(
        _df, drop_sensitive_features=drop_sensitive, fast=fast,
        c_fn=c_fn, c_fp=c_fp, test_size=test_size,
        progress=lambda p, t: bar.progress(p, text=t),
    )
    bar.empty()
    return exp


@st.cache_resource(show_spinner="กำลังคำนวณ permutation importance บนชุดทดสอบ ...")
def get_perm(_exp, key, model_name, n_repeats):
    r = _exp["results"][model_name]
    return cc.perm_importance(r["best_estimator"], _exp["X_test"], _exp["y_test"], n_repeats)


@st.cache_resource(show_spinner="กำลังคำนวณ SHAP บนชุดทดสอบ ...")
def get_shap(_exp, key, model_name):
    r = _exp["results"][model_name]
    return cc.shap_values(r["best_estimator"], _exp["X_train"], _exp["X_test"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("💳 German Credit Risk")
st.sidebar.caption("Statlog German Credit · OpenML data_id=31")

source = st.sidebar.radio("แหล่งข้อมูล", ["OpenML (data_id=31)", "อัปโหลด CSV"],
                          help="ถ้าเซิร์ฟเวอร์เข้าถึง OpenML ไม่ได้ ให้อัปโหลดไฟล์ CSV ที่มีคอลัมน์เดียวกัน")
df = None
if source.startswith("OpenML"):
    try:
        df = load_openml()
        data_key = "openml31"
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"โหลดจาก OpenML ไม่สำเร็จ: {e}")
else:
    up = st.sidebar.file_uploader("ไฟล์ CSV (ต้องมีคอลัมน์ 'class')", type="csv")
    if up is not None:
        df = load_csv(up)
        data_key = f"csv-{up.name}-{up.size}"

if df is None:
    st.info("กรุณาเลือกแหล่งข้อมูลจากแถบด้านซ้าย")
    st.stop()

st.sidebar.markdown("### ⚙️ ตั้งค่าการทดลอง")
c_fn = st.sidebar.number_input("ต้นทุน FN (อนุมัติคน bad)", 0.1, 100.0, 5.0, 0.5)
c_fp = st.sidebar.number_input("ต้นทุน FP (ปฏิเสธคน good)", 0.1, 100.0, 1.0, 0.5)
test_size = st.sidebar.slider("สัดส่วนชุดทดสอบ", 0.15, 0.40, 0.20, 0.05)
drop_sensitive = st.sidebar.checkbox(
    "ตัด personal_status และ age ออกจากฟีเจอร์",
    help="Fairness through unawareness — ใช้เปรียบเทียบผลต่อ fairness metrics",
)
fast = st.sidebar.checkbox("โหมดเร็ว (grid เล็กลง)", value=False)

exp = train(df, data_key, drop_sensitive, fast, c_fn, c_fp, test_size)
exp_key = (data_key, drop_sensitive, fast, c_fn, c_fp, test_size)
results = exp["results"]
model_names = list(results)
sel = st.sidebar.selectbox(
    "โมเดลที่ใช้วิเคราะห์", model_names, index=model_names.index(exp["best_name"]),
    help="ค่าเริ่มต้นคือโมเดลที่มี OOF expected cost ต่ำสุดบนชุดฝึก",
)
R = results[sel]
st.sidebar.success(f"โมเดลที่ดีที่สุด (OOF cost): **{exp['best_name']}**")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("การประเมินความเสี่ยงสินเชื่อ: โมเดล · การอธิบาย · ความเป็นธรรม")
c1, c2, c3, c4 = st.columns(4)
c1.metric("จำนวนตัวอย่าง", f"{len(df):,}")
c2.metric("สัดส่วน bad", f"{(df[cc.TARGET].astype(str) == 'bad').mean():.1%}")
c3.metric(f"Threshold ({sel})", f"{R['threshold']:.2f}")
c4.metric("Test expected cost", f"{R['test_metrics']['Expected cost']:.3f}",
          delta=f"{R['test_metrics']['Expected cost'] - R['test_metrics_05']['Expected cost']:+.3f} vs @0.5",
          delta_color="inverse")

tabs = st.tabs(["📊 EDA", "🤖 โมเดล", "💰 Threshold", "🔍 การอธิบาย", "⚖️ ความเป็นธรรม", "🧮 ทำนายรายบุคคล"])

# ---------------------------------------------------------------------------
# 1) EDA
# ---------------------------------------------------------------------------
with tabs[0]:
    eda = cc.eda_summary(df)
    dfa = cc.add_sensitive_attributes(df)
    st.subheader("ภาพรวมข้อมูล")
    a, b, c = st.columns(3)
    a.metric("แถว × คอลัมน์", f"{eda['shape'][0]} × {eda['shape'][1]}")
    b.metric("ฟีเจอร์ตัวเลข / หมวดหมู่", f"{len(eda['num_cols'])} / {len(eda['cat_cols'])}")
    c.metric("ค่าที่หายไปทั้งหมด", int(eda["missing"].sum()))
    st.dataframe(df.head(20), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        cc_ = eda["class_counts"].reset_index()
        cc_.columns = ["class", "count"]
        st.plotly_chart(px.bar(cc_, x="class", y="count", color="class", text="count",
                               title="การกระจายของ target",
                               color_discrete_map={"good": "#2a9d8f", "bad": "#e76f51"}),
                        width="stretch")
    with col2:
        st.markdown("**สถิติเชิงพรรณนา (ตัวเลข)**")
        st.dataframe(eda["describe"].round(2), width="stretch")

    st.subheader("ฟีเจอร์ตัวเลขแยกตาม class")
    num = st.selectbox("เลือกฟีเจอร์ตัวเลข", eda["num_cols"], index=eda["num_cols"].index("credit_amount")
                       if "credit_amount" in eda["num_cols"] else 0)
    st.plotly_chart(px.histogram(df, x=num, color=cc.TARGET, marginal="box", barmode="overlay",
                                 opacity=0.65, color_discrete_map={"good": "#2a9d8f", "bad": "#e76f51"}),
                    width="stretch")

    st.subheader("อัตรา bad ในแต่ละหมวดหมู่")
    cat = st.selectbox("เลือกฟีเจอร์หมวดหมู่", eda["cat_cols"] + ["sex", "age_group"])
    rate = (dfa.assign(bad=(dfa[cc.TARGET].astype(str) == "bad").astype(int))
            .groupby(cat, observed=True)["bad"].agg(["mean", "count"]).reset_index())
    fig = px.bar(rate, x=cat, y="mean", text=rate["count"].map(lambda v: f"n={v}"),
                 labels={"mean": "bad rate"}, color="mean", color_continuous_scale="OrRd")
    fig.add_hline(y=(dfa[cc.TARGET].astype(str) == "bad").mean(), line_dash="dash",
                  annotation_text="ค่าเฉลี่ยรวม")
    st.plotly_chart(fig, width="stretch")

    st.subheader("ความสัมพันธ์ระหว่างฟีเจอร์ตัวเลข")
    st.plotly_chart(px.imshow(df[eda["num_cols"]].corr().round(2), text_auto=True,
                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                    width="stretch")

# ---------------------------------------------------------------------------
# 2) Models
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("เปรียบเทียบ 3 อัลกอริทึม")
    st.caption("ปรับ hyperparameter ด้วย GridSearchCV + StratifiedKFold(k=5) บนชุดฝึกเท่านั้น "
               "(scoring = ROC-AUC) · threshold เลือกจาก out-of-fold predictions · "
               "ชุดทดสอบใช้ประเมินผลครั้งสุดท้ายเท่านั้น")
    tbl = cc.comparison_table(exp)
    st.dataframe(tbl.style.format("{:.3f}")
                 .highlight_min(subset=["OOF expected cost", "Test expected cost"], color="#c7f0d8")
                 .highlight_max(subset=["CV ROC-AUC (train)", "Test ROC-AUC"], color="#c7f0d8"),
                 width="stretch")

    st.markdown("**Hyperparameters ที่ดีที่สุด**")
    for n, r in results.items():
        st.write(f"- **{n}**: `{r['best_params']}`")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for n, r in results.items():
            fpr, tpr, _ = roc_curve(exp["y_test"], r["test_proba"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{n} (AUC={r['test_metrics']['ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), showlegend=False))
        fig.update_layout(title="ROC curve (ชุดทดสอบ)", xaxis_title="FPR", yaxis_title="TPR",
                          legend=dict(x=0.4, y=0.05))
        st.plotly_chart(fig, width="stretch")
    with col2:
        m = R["test_metrics"]
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["ทาย good", "ทาย bad"], y=["จริง good", "จริง bad"],
                        title=f"Confusion matrix — {sel} @ {R['threshold']:.2f}")
        st.plotly_chart(fig, width="stretch")

    st.markdown(f"**ตัวชี้วัดบนชุดทดสอบ — {sel}**")
    st.dataframe(pd.DataFrame({"@ cost-optimal threshold": R["test_metrics"],
                               "@ 0.5": R["test_metrics_05"]}).T.round(3), width="stretch")

# ---------------------------------------------------------------------------
# 3) Threshold
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader(f"เลือก threshold ด้วย expected cost (FN:FP = {c_fn:g}:{c_fp:g})")
    st.latex(r"\text{Expected cost}(t)=\frac{C_{FN}\cdot FN(t)+C_{FP}\cdot FP(t)}{n}")
    st.caption("คำนวณจาก out-of-fold probabilities ของชุดฝึก (5 folds) — ไม่ใช้ชุดทดสอบในการเลือก")
    fig = go.Figure()
    for n, r in results.items():
        fig.add_trace(go.Scatter(x=r["oof_curve"]["threshold"], y=r["oof_curve"]["expected_cost"], name=n))
        fig.add_trace(go.Scatter(x=[r["threshold"]], y=[r["oof_cost"]], mode="markers",
                                 marker=dict(size=12, symbol="star"), showlegend=False,
                                 hovertext=f"{n}: t={r['threshold']:.2f}"))
    fig.add_vline(x=c_fp / (c_fp + c_fn), line_dash="dot",
                  annotation_text=f"ทฤษฎี (calibrated) = {c_fp / (c_fp + c_fn):.3f}")
    fig.update_layout(xaxis_title="Threshold", yaxis_title="Expected cost / ราย (OOF)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame({n: {"threshold": r["threshold"], "OOF cost": r["oof_cost"],
                                   "Test cost": r["test_metrics"]["Expected cost"],
                                   "Test cost @0.5": r["test_metrics_05"]["Expected cost"]}
                               for n, r in results.items()}).T.round(3), width="stretch")

    st.markdown("#### 🔧 What-if: ลองเลื่อน threshold บนชุดทดสอบ")
    t = st.slider("Threshold", 0.01, 0.99, float(R["threshold"]), 0.01)
    mt = cc.classification_metrics(exp["y_test"], R["test_proba"], t, c_fn, c_fp)
    k = st.columns(5)
    k[0].metric("Expected cost", f"{mt['Expected cost']:.3f}")
    k[1].metric("Recall (bad)", f"{mt['Recall']:.2f}")
    k[2].metric("Precision (bad)", f"{mt['Precision']:.2f}")
    k[3].metric("อัตราอนุมัติ", f"{(mt['TN'] + mt['FN']) / len(exp['y_test']):.1%}")
    k[4].metric("FN / FP", f"{mt['FN']} / {mt['FP']}")

# ---------------------------------------------------------------------------
# 4) Explainability
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader(f"Permutation importance บนชุดทดสอบ — {sel}")
    st.caption("วัดว่า ROC-AUC ลดลงเท่าใดเมื่อสลับค่าคอลัมน์นั้นแบบสุ่ม (ระดับคอลัมน์ดิบ ก่อน one-hot)")
    n_rep = st.slider("จำนวนรอบ (n_repeats)", 5, 50, 20, 5)
    pi = get_perm(exp, exp_key, sel, n_rep)
    fig = px.bar(pi.sort_values("importance_mean"), x="importance_mean", y="feature",
                 error_x="importance_std", orientation="h", height=600,
                 labels={"importance_mean": "ΔROC-AUC เมื่อ permute"})
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"SHAP บนชุดทดสอบ — {sel}")
    st.caption("ค่า SHAP ในพื้นที่ฟีเจอร์หลัง preprocessing · ค่าบวก = ผลักไปทาง 'bad'")
    try:
        sv, mean_abs = get_shap(exp, exp_key, sel)
        top = st.slider("จำนวนฟีเจอร์ที่แสดง", 5, 30, 15)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(mean_abs.head(top).sort_values("mean_abs_shap"), x="mean_abs_shap",
                         y="feature", orientation="h", height=550, title="mean(|SHAP|)")
            st.plotly_chart(fig, width="stretch")
        with col2:
            import shap
            plt.figure()
            shap.plots.beeswarm(sv, max_display=top, show=False)
            fig = plt.gcf()
            fig.set_size_inches(7, 0.35 * top + 1.5)
            st.pyplot(fig, clear_figure=True)
        st.markdown("**อธิบายรายบุคคล (waterfall)**")
        i = st.number_input("ลำดับตัวอย่างในชุดทดสอบ (ที่สุ่มมาคำนวณ SHAP)", 0, len(sv) - 1, 0)
        plt.figure()
        shap.plots.waterfall(sv[int(i)], max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f"คำนวณ SHAP ไม่สำเร็จ: {e}")

# ---------------------------------------------------------------------------
# 5) Fairness
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader(f"Fairness metrics บนชุดทดสอบ — {sel} @ threshold {R['threshold']:.2f}")
    st.caption("positive = 'bad' = ถูกปฏิเสธสินเชื่อ · sex มาจาก personal_status · "
               f"age_group แบ่งเป็น {', '.join(cc.AGE_LABELS)}")
    fr = cc.fairness_report(exp["y_test"], R["test_proba"], exp["g_test"], R["threshold"])
    attr = st.radio("กลุ่มที่ต้องการวิเคราะห์", cc.SENSITIVE_COLS, horizontal=True)
    gm, summ = fr[attr]["by_group"], fr[attr]["summary"]

    k = st.columns(4)
    di = summ["Disparate impact ratio (approval, min/max)"]
    k[0].metric("Disparate impact (approval)", f"{di:.3f}",
                delta="ผ่านเกณฑ์ 4/5" if di >= 0.8 else "ต่ำกว่า 0.8", delta_color="normal" if di >= 0.8 else "inverse")
    k[1].metric("Demographic parity diff.", f"{summ['Demographic parity difference (selection rate)']:.3f}")
    k[2].metric("Equal opportunity diff. (TPR)", f"{summ['Equal opportunity difference (TPR)']:.3f}")
    k[3].metric("Equalized odds diff.", f"{summ['Equalized odds difference (max of TPR, FPR)']:.3f}")

    st.dataframe(gm.style.format({c: "{:.3f}" for c in gm.columns if c != "n"}), width="stretch")
    long = gm[["base_rate(bad)", "selection_rate", "TPR", "FPR", "Precision", "Accuracy"]].reset_index() \
        .melt(id_vars="group", var_name="metric", value_name="value")
    st.plotly_chart(px.bar(long, x="metric", y="value", color="group", barmode="group",
                           title=f"ตัวชี้วัดรายกลุ่มตาม {attr}"), width="stretch")
    st.markdown("**สรุปช่องว่างระหว่างกลุ่ม (max − min, ratio = min/max)**")
    st.dataframe(pd.Series(summ, name="value").round(4).to_frame(), width="stretch")

    with st.expander("เปรียบเทียบทุกโมเดล"):
        rows = []
        for n, r in results.items():
            f = cc.fairness_report(exp["y_test"], r["test_proba"], exp["g_test"], r["threshold"])
            for a in cc.SENSITIVE_COLS:
                rows.append({"model": n, "attribute": a, **f[a]["summary"]})
        st.dataframe(pd.DataFrame(rows).set_index(["model", "attribute"]).round(3), width="stretch")

    with st.expander("ความหมายของตัวชี้วัด"):
        st.markdown("""
- **Selection rate**: สัดส่วนที่โมเดลทำนายว่า bad (ถูกปฏิเสธ) — ใช้วัด *demographic parity*
- **Disparate impact ratio**: อัตราอนุมัติของกลุ่มต่ำสุด ÷ กลุ่มสูงสุด (กฎ 4/5 → ควร ≥ 0.8)
- **Equal opportunity difference**: ความต่างของ TPR (จับคน bad ได้เท่ากันไหม)
- **Equalized odds difference**: ค่าสูงสุดของความต่าง TPR และ FPR (FPR สูง = คนดีถูกปฏิเสธมาก)
- **Predictive parity**: ความต่างของ Precision ระหว่างกลุ่ม
- ชุดทดสอบมีขนาดเล็ก (โดยเฉพาะกลุ่ม ≤25 ปี / หญิง) ตัวเลขจึงแปรปรวนสูง ควรตีความร่วมกับขนาดกลุ่ม `n`
""")

# ---------------------------------------------------------------------------
# 6) Single prediction
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader(f"ทำนายความเสี่ยงของผู้ขอสินเชื่อ — {sel}")
    X_ref = exp["X_train"]
    with st.form("predict"):
        cols = st.columns(3)
        row = {}
        for j, c in enumerate(X_ref.columns):
            with cols[j % 3]:
                if pd.api.types.is_numeric_dtype(X_ref[c]):
                    lo, hi, med = float(X_ref[c].min()), float(X_ref[c].max()), float(X_ref[c].median())
                    row[c] = st.number_input(c, lo, hi, med)
                else:
                    opts = sorted(X_ref[c].astype(str).unique())
                    row[c] = st.selectbox(c, opts)
        submitted = st.form_submit_button("ทำนาย")
    if submitted:
        x = pd.DataFrame([row])
        for c in X_ref.columns:
            if isinstance(X_ref[c].dtype, pd.CategoricalDtype):
                x[c] = pd.Categorical(x[c], categories=X_ref[c].cat.categories)
        p = float(R["best_estimator"].predict_proba(x)[:, 1][0])
        bad = p >= R["threshold"]
        a, b = st.columns(2)
        a.metric("P(bad)", f"{p:.1%}")
        b.metric("ผลการพิจารณา", "❌ ปฏิเสธ (bad)" if bad else "✅ อนุมัติ (good)")
        st.progress(min(p, 1.0), text=f"threshold = {R['threshold']:.2f}")

st.sidebar.markdown("---")
st.sidebar.caption("random_state=42 · StratifiedKFold(k=5) · positive class = 'bad'")
