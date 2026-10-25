"""
credit_core.py
Logic ทั้งหมดของโปรเจกต์ (แยกจาก UI เพื่อให้ทดสอบ/นำไปใช้ซ้ำได้)
- โหลดข้อมูล Statlog German Credit (OpenML data_id=31)
- สร้าง target (positive = 'bad'), split แบบ stratified
- Pipeline (ColumnTransformer + model) 3 อัลกอริทึม
- Tuning ด้วย StratifiedKFold(k=5) บนชุดฝึกเท่านั้น
- เลือก threshold จาก out-of-fold predictions ให้ expected cost ต่ำสุด (FN:FP = 5:1)
- Permutation importance และ SHAP บนชุดทดสอบ
- Fairness metrics ตาม sex และ age_group
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
N_JOBS = 2  # Streamlit Cloud มี CPU/RAM จำกัด (~1 GB) — อย่าใช้ -1
TARGET = "class"
POS_LABEL = "bad"
COST_FN = 5.0   # อนุมัติให้คนที่จะผิดนัด (bad ถูกทายเป็น good) — แพงกว่า
COST_FP = 1.0   # ปฏิเสธคนดี (good ถูกทายเป็น bad)
SENSITIVE_COLS = ["sex", "age_group"]
AGE_BINS = [0, 25, 40, 200]
AGE_LABELS = ["<=25", "26-40", ">40"]


# ---------------------------------------------------------------------------
# 1) Data
# ---------------------------------------------------------------------------
def load_german_credit() -> pd.DataFrame:
    """ดาวน์โหลดชุดข้อมูลจาก OpenML (data_id=31)"""
    from sklearn.datasets import fetch_openml
    return fetch_openml(data_id=31, as_frame=True).frame


def add_sensitive_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """สร้างคอลัมน์ sex (จาก personal_status) และ age_group (จาก age)"""
    out = df.copy()
    ps = out["personal_status"].astype(str).str.lower()
    out["sex"] = np.where(ps.str.startswith("female"), "female", "male")
    out["age_group"] = pd.cut(
        out["age"].astype(float), bins=AGE_BINS, labels=AGE_LABELS, right=True
    ).astype(str)
    return out


def make_xy(df: pd.DataFrame, drop_sensitive_features: bool = False):
    """
    คืนค่า X, y, groups
    - y = 1 ถ้า class == 'bad' (positive class)
    - groups = DataFrame ของ sex, age_group (ใช้วัด fairness ไม่ใช้เป็นฟีเจอร์ซ้ำ)
    - drop_sensitive_features=True จะตัด personal_status และ age ออกจากฟีเจอร์
      (fairness through unawareness) เพื่อเปรียบเทียบ
    """
    d = add_sensitive_attributes(df)
    y = (d[TARGET].astype(str) == POS_LABEL).astype(int)
    groups = d[SENSITIVE_COLS].copy()
    X = d.drop(columns=[TARGET] + SENSITIVE_COLS)
    if drop_sensitive_features:
        X = X.drop(columns=["personal_status", "age"])
    return X, y, groups


def split_data(X, y, groups, test_size: float = 0.2):
    return train_test_split(
        X, y, groups, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def eda_summary(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != TARGET]
    return {
        "shape": df.shape,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "missing": df.isna().sum(),
        "class_counts": df[TARGET].value_counts(),
        "describe": df[num_cols].describe().T,
    }


# ---------------------------------------------------------------------------
# 2) Pipelines + tuning
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        verbose_feature_names_out=True,
    )


def model_specs(fast: bool = False) -> dict:
    """อัลกอริทึม 3 ตัว + พารามิเตอร์ที่จะค้นหา (fast=True ใช้ grid เล็กลง)"""
    specs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [300],
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 5],
                "model__max_features": ["sqrt", 0.5],
                "model__class_weight": [None, "balanced_subsample"],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }
    if fast:
        specs["Logistic Regression"][1].update({"model__C": [0.1, 1.0]})
        specs["Random Forest"][1].update({
            "model__n_estimators": [200], "model__max_depth": [None, 8],
            "model__max_features": ["sqrt"],
        })
        specs["Gradient Boosting"][1].update({
            "model__n_estimators": [150], "model__subsample": [0.8],
        })
    return specs


def tune_model(name, estimator, grid, X_train, y_train, scoring="roc_auc"):
    """GridSearchCV + StratifiedKFold(5) บนชุดฝึก แล้วสร้าง OOF probabilities"""
    pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", estimator)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=N_JOBS, refit=True)
    gs.fit(X_train, y_train)
    # OOF predictions ด้วยพารามิเตอร์ที่ดีที่สุด (ใช้ fold ชุดเดิม ไม่แตะ test)
    oof = cross_val_predict(
        gs.best_estimator_, X_train, y_train, cv=cv, method="predict_proba", n_jobs=N_JOBS
    )[:, 1]
    return {
        "name": name,
        "search": gs,
        "best_estimator": gs.best_estimator_,
        "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
        "cv_auc": gs.best_score_,
        "cv_auc_std": gs.cv_results_["std_test_score"][gs.best_index_],
        "oof_proba": oof,
    }


# ---------------------------------------------------------------------------
# 2.4) Cost-sensitive threshold
# ---------------------------------------------------------------------------
def expected_cost(y_true, proba, threshold, c_fn=COST_FN, c_fp=COST_FP) -> float:
    """ค่าใช้จ่ายเฉลี่ยต่อรายการ = (c_fn*FN + c_fp*FP) / n"""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (pred == 0))
    fp = np.sum((y_true == 0) & (pred == 1))
    return (c_fn * fn + c_fp * fp) / len(y_true)


def cost_curve(y_true, proba, c_fn=COST_FN, c_fp=COST_FP, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2) if grid is None else grid
    return pd.DataFrame({
        "threshold": grid,
        "expected_cost": [expected_cost(y_true, proba, t, c_fn, c_fp) for t in grid],
    })


def best_threshold(y_true, proba, c_fn=COST_FN, c_fp=COST_FP):
    curve = cost_curve(y_true, proba, c_fn, c_fp)
    row = curve.loc[curve["expected_cost"].idxmin()]
    return float(row["threshold"]), float(row["expected_cost"]), curve


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def classification_metrics(y_true, proba, threshold, c_fn=COST_FN, c_fp=COST_FP) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Expected cost": (c_fn * fn + c_fp * fp) / len(pred),
    }


def run_experiment(df: pd.DataFrame, drop_sensitive_features=False, fast=False,
                   c_fn=COST_FN, c_fp=COST_FP, test_size=0.2, progress=None) -> dict:
    """รันทั้งกระบวนการ ตั้งแต่ split จนถึงประเมินบน test"""
    X, y, groups = make_xy(df, drop_sensitive_features)
    X_tr, X_te, y_tr, y_te, g_tr, g_te = split_data(X, y, groups, test_size)

    results = {}
    specs = model_specs(fast)
    for i, (name, (est, grid)) in enumerate(specs.items()):
        if progress:
            progress(i / len(specs), f"กำลังปรับจูน {name} ...")
        r = tune_model(name, est, grid, X_tr, y_tr)
        thr, oof_cost, curve = best_threshold(y_tr, r["oof_proba"], c_fn, c_fp)
        r.update({"threshold": thr, "oof_cost": oof_cost, "oof_curve": curve})
        test_proba = r["best_estimator"].predict_proba(X_te)[:, 1]
        r["test_proba"] = test_proba
        r["test_metrics"] = classification_metrics(y_te, test_proba, thr, c_fn, c_fp)
        r["test_metrics_05"] = classification_metrics(y_te, test_proba, 0.5, c_fn, c_fp)
        results[name] = r
    if progress:
        progress(1.0, "เสร็จสิ้น")

    # เลือกโมเดลจาก OOF cost บนชุดฝึก (ไม่ใช้ test ในการเลือก)
    best_name = min(results, key=lambda n: (results[n]["oof_cost"], -results[n]["cv_auc"]))
    return {
        "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "g_train": g_tr, "g_test": g_te, "results": results, "best_name": best_name,
        "c_fn": c_fn, "c_fp": c_fp, "drop_sensitive_features": drop_sensitive_features,
    }


def comparison_table(exp: dict) -> pd.DataFrame:
    rows = []
    for name, r in exp["results"].items():
        m = r["test_metrics"]
        rows.append({
            "Model": name,
            "CV ROC-AUC (train)": r["cv_auc"],
            "CV AUC std": r["cv_auc_std"],
            "Threshold (OOF)": r["threshold"],
            "OOF expected cost": r["oof_cost"],
            "Test ROC-AUC": m["ROC-AUC"],
            "Test PR-AUC": m["PR-AUC"],
            "Test Recall(bad)": m["Recall"],
            "Test Precision(bad)": m["Precision"],
            "Test expected cost": m["Expected cost"],
            "Test cost @0.5": r["test_metrics_05"]["Expected cost"],
        })
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# 3) Explainability
# ---------------------------------------------------------------------------
def perm_importance(model, X_test, y_test, n_repeats=20, scoring="roc_auc") -> pd.DataFrame:
    """Permutation importance ระดับคอลัมน์ดิบ บนชุดทดสอบ"""
    pi = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, scoring=scoring,
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    )
    return (pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True))


def shap_values(model: Pipeline, X_train, X_test, max_rows=300):
    """
    คำนวณ SHAP บนชุดทดสอบ (ในพื้นที่ฟีเจอร์หลัง preprocess) สำหรับ class 'bad'
    คืนค่า (shap.Explanation, mean|SHAP| DataFrame)
    """
    import shap

    prep, est = model.named_steps["prep"], model.named_steps["model"]
    names = [n.replace("num__", "").replace("cat__", "") for n in prep.get_feature_names_out()]
    Xt_test = pd.DataFrame(prep.transform(X_test), columns=names, index=X_test.index)
    if len(Xt_test) > max_rows:
        Xt_test = Xt_test.sample(max_rows, random_state=RANDOM_STATE)
    Xt_bg = pd.DataFrame(prep.transform(X_train), columns=names)
    Xt_bg = Xt_bg.sample(min(200, len(Xt_bg)), random_state=RANDOM_STATE)

    if isinstance(est, LogisticRegression):
        explainer = shap.LinearExplainer(
            est, shap.maskers.Independent(Xt_bg, max_samples=len(Xt_bg))
        )
    else:
        explainer = shap.TreeExplainer(est)
    exp = explainer(Xt_test)
    if exp.values.ndim == 3:            # (n, features, classes) -> class 1
        exp = exp[:, :, 1]
    mean_abs = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(exp.values).mean(0)})
                .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    return exp, mean_abs


# ---------------------------------------------------------------------------
# 4) Fairness
# ---------------------------------------------------------------------------
def group_metrics(y_true, proba, groups: pd.Series, threshold: float) -> pd.DataFrame:
    """
    ตัวชี้วัดรายกลุ่ม (positive = bad = ถูกปฏิเสธสินเชื่อ)
    - selection_rate  : สัดส่วนที่ถูกทำนายว่า bad (ถูกปฏิเสธ)
    - approval_rate   : 1 - selection_rate
    - TPR / FNR / FPR / Precision / Accuracy / AUC
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    g = np.asarray(groups).astype(str)
    rows = []
    for level in sorted(np.unique(g)):
        m = g == level
        yt, yp, pp = y_true[m], pred[m], proba[m]
        tp = np.sum((yt == 1) & (yp == 1)); fn = np.sum((yt == 1) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1)); tn = np.sum((yt == 0) & (yp == 0))
        rows.append({
            "group": level, "n": int(m.sum()),
            "base_rate(bad)": yt.mean(),
            "selection_rate": yp.mean(),
            "approval_rate": 1 - yp.mean(),
            "TPR": tp / (tp + fn) if tp + fn else np.nan,
            "FNR": fn / (tp + fn) if tp + fn else np.nan,
            "FPR": fp / (fp + tn) if fp + tn else np.nan,
            "Precision": tp / (tp + fp) if tp + fp else np.nan,
            "Accuracy": (tp + tn) / m.sum(),
            "AUC": roc_auc_score(yt, pp) if len(np.unique(yt)) == 2 else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def fairness_summary(gm: pd.DataFrame) -> dict:
    """สรุปความต่างระหว่างกลุ่ม (max - min) และอัตราส่วน (min / max)"""
    def diff(c): return float(gm[c].max() - gm[c].min())
    sr = gm["selection_rate"]; ar = gm["approval_rate"]
    return {
        "Demographic parity difference (selection rate)": diff("selection_rate"),
        "Disparate impact ratio (approval, min/max)": float(ar.min() / ar.max()) if ar.max() > 0 else np.nan,
        "Equal opportunity difference (TPR)": diff("TPR"),
        "FPR difference": diff("FPR"),
        "Equalized odds difference (max of TPR, FPR)": max(diff("TPR"), diff("FPR")),
        "Predictive parity difference (Precision)": diff("Precision"),
        "Accuracy difference": diff("Accuracy"),
    }


def fairness_report(y_test, proba, g_test: pd.DataFrame, threshold: float) -> dict:
    out = {}
    for col in SENSITIVE_COLS:
        gm = group_metrics(y_test, proba, g_test[col], threshold)
        out[col] = {"by_group": gm, "summary": fairness_summary(gm)}
    return out
