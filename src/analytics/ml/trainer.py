"""Huấn luyện & đánh giá mô hình Machine Learning có kiểm soát rò rỉ dữ liệu (data leakage).

Quy trình chuẩn:
1. Xác định loại bài toán (hồi quy / phân loại) → người dùng xác nhận.
2. Loại cột ID, văn bản dài, cột trùng thông tin với biến mục tiêu (nghi rò rỉ).
3. Chia train/test TRƯỚC mọi bước học tham số (stratify cho phân loại).
4. Tiền xử lý nằm TRONG Pipeline (Imputer, Scaler, One-Hot chỉ fit trên train) ⇒ không rò rỉ.
5. Mất cân bằng lớp ⇒ class_weight='balanced' / sample_weight.
6. So sánh với baseline (DummyRegressor/DummyClassifier), cross-validation 5-fold trên tập train.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight

from src.analytics.config import CV_FOLDS, MAX_TRAIN_ROWS, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.analytics.data import schema

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover - phụ thuộc môi trường
    HAS_XGB = False

LINEAR_MODELS = {"Linear Regression", "Logistic Regression"}
BASELINE_NAME = "Baseline (Dummy)"


# ------------------------------------------------------------------ Loại bài toán
def detect_problem_type(y: pd.Series) -> tuple[str, str]:
    y = y.dropna()
    if y.empty:
        return "regression", "Biến mục tiêu không có giá trị."
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification", f"Biến mục tiêu dạng phân loại ({y.nunique()} lớp)."
    nun = y.nunique()
    is_int = bool(np.all(np.mod(y, 1) == 0))
    if nun <= 2 or (is_int and nun <= 10):
        return "classification", f"Biến số nguyên với {nun} giá trị khác nhau ⇒ xem như phân loại."
    return "regression", f"Biến số liên tục với {nun:,} giá trị khác nhau ⇒ hồi quy."


# ------------------------------------------------------------------ Chọn đặc trưng
def usable_features(df: pd.DataFrame, target: str, max_card: int = 60) -> tuple[list[str], dict[str, str]]:
    """Trả về (đặc trưng dùng được, {cột bị loại: lý do})."""
    excluded: dict[str, str] = {}
    keep = []
    profiles = {p.name: p for p in schema.profile_columns(df)}
    for c in df.columns:
        if c == target:
            continue
        p = profiles[c]
        if p.role == "id":
            excluded[c] = "Cột định danh (ID) - không mang thông tin dự đoán tổng quát."
        elif p.role == "constant":
            excluded[c] = "Hằng số - không có biến thiên."
        elif p.role in ("text",):
            excluded[c] = "Văn bản dài - cần text mining riêng, không đưa trực tiếp vào mô hình."
        elif p.role == "datetime":
            excluded[c] = "Thời gian - hãy tách năm/tháng ở bước làm sạch nếu cần."
        elif p.role in ("categorical", "geo") and p.n_unique > max_card:
            excluded[c] = f"Phân loại có {p.n_unique:,} giá trị (> {max_card}) - One-Hot sẽ quá thưa."
        elif p.missing_pct > 60:
            excluded[c] = f"Thiếu {p.missing_pct:.0f}% giá trị."
        else:
            keep.append(c)
    return keep, excluded


def leakage_check(df: pd.DataFrame, features: list[str], target: str, threshold: float = 0.95) -> dict[str, str]:
    """Phát hiện đặc trưng gần như trùng thông tin với biến mục tiêu (dấu hiệu rò rỉ)."""
    sus: dict[str, str] = {}
    y = df[target]
    tname = schema.norm_name(target)
    if detect_problem_type(y)[0] == "classification":
        # một biến số đơn lẻ tách hoàn hảo các lớp (cây độ sâu 2 đạt ≥ 99%) ⇒ nhiều khả năng mục tiêu được suy ra từ biến đó
        sample = df[[target] + [c for c in features if pd.api.types.is_numeric_dtype(df[c])]].dropna(subset=[target])
        sample = sample.sample(min(len(sample), 5000), random_state=RANDOM_STATE) if len(sample) else sample
        for c in sample.columns.drop(target):
            x = sample[[c]].fillna(sample[c].median())
            if len(x) >= 30 and x[c].nunique() > 1 and y.nunique() >= 2:
                acc = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE).fit(x, sample[target].astype(str)).score(
                    x, sample[target].astype(str))
                if acc >= 0.99:
                    sus[c] = f"Một mình biến này phân loại đúng {acc * 100:.1f}% mẫu (cây độ sâu 2)"
    for c in features:
        if c in sus:
            continue
        x = df[c]
        try:
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                pair = pd.concat([x, y], axis=1).dropna()
                if len(pair) > 10 and pair.iloc[:, 0].nunique() > 1:
                    r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                    if abs(r) >= threshold:
                        sus[c] = f"|r| = {abs(r):.3f} với biến mục tiêu"
                        continue
            stem = tname.split(" ")[0]
            if len(stem) >= 4 and stem in schema.norm_name(c):
                sus[c] = "Tên cột cùng gốc với biến mục tiêu (có thể được suy ra từ mục tiêu)"
                continue
            if not pd.api.types.is_numeric_dtype(x):
                g = pd.concat([x.astype(str), y], axis=1).dropna().groupby(x.name)[target].nunique()
                if len(g) > 5 and (g == 1).mean() > 0.98 and x.nunique() > 0.5 * y.nunique():
                    sus[c] = "Mỗi giá trị của cột xác định duy nhất biến mục tiêu"
        except Exception:
            continue
    return sus


# ------------------------------------------------------------------ Mô hình
def model_catalog(problem: str, balanced: bool) -> dict:
    cw = "balanced" if balanced else None
    if problem == "regression":
        cat = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.1,
                                                                     subsample=0.8, random_state=RANDOM_STATE),
        }
        if HAS_XGB:
            cat["XGBoost Regressor"] = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8,
                                                    colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        return cat
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=cw),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, class_weight=cw, random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight=cw, n_jobs=-1,
                                                           random_state=RANDOM_STATE),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                                                                   subsample=0.8, random_state=RANDOM_STATE),
    }


def build_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    try:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False)
    except TypeError:  # sklearn cũ
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformers = []
    if num:
        transformers.append(("num", Pipeline(num_steps), num))
    if cat:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]), cat))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


@dataclass
class TrainResult:
    problem: str
    target: str
    features: list[str]
    comparison: pd.DataFrame
    best_name: str
    best_reason: str
    details: dict = field(default_factory=dict)       # tên mô hình → metrics/confusion/importance
    pipelines: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    classes: list = field(default_factory=list)
    saved_path: str | None = None
    baseline: dict = field(default_factory=dict)


def _regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    out = {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}
    nz = np.abs(np.asarray(y_true)) > 1e-9
    out["MAPE (%)"] = float(np.mean(np.abs((np.asarray(y_true)[nz] - np.asarray(y_pred)[nz]) / np.asarray(y_true)[nz])) * 100) if nz.any() else np.nan
    return {k: float(v) for k, v in out.items()}


def _classification_metrics(y_true, y_pred, proba, classes) -> dict:
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC": np.nan,
    }
    if proba is not None:
        try:
            if len(classes) == 2:
                out["ROC-AUC"] = roc_auc_score(y_true, proba[:, 1])
            elif len(set(y_true)) == len(classes):
                out["ROC-AUC"] = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes)
        except ValueError:
            pass
    return {k: float(v) for k, v in out.items()}


def _importance(pipe: Pipeline) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    try:
        names = pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        return pd.DataFrame()
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.atleast_2d(model.coef_)
        vals = np.abs(coef).mean(axis=0)
    else:
        return pd.DataFrame()
    if len(vals) != len(names):
        return pd.DataFrame()
    imp = pd.DataFrame({"Đặc trưng": names, "Mức quan trọng": vals}).sort_values("Mức quan trọng", ascending=False)
    imp["Mức quan trọng"] = imp["Mức quan trọng"] / (imp["Mức quan trọng"].sum() or 1)
    return imp.head(25).reset_index(drop=True)


def train_models(df: pd.DataFrame, target: str, features: list[str], problem: str | None = None,
                 model_names: list[str] | None = None, test_size: float = TEST_SIZE, handle_imbalance: str = "auto",
                 max_rows: int = MAX_TRAIN_ROWS, cv: bool = True, progress=None) -> TrainResult:
    if target in features:
        raise ValueError("Biến mục tiêu không được nằm trong danh sách đặc trưng.")
    if not features:
        raise ValueError("Cần chọn ít nhất một đặc trưng.")
    notes: list[str] = []
    data = df[features + [target]].copy()
    n_missing_target = int(data[target].isna().sum())
    data = data.dropna(subset=[target])
    if n_missing_target:
        notes.append(f"Loại {n_missing_target:,} dòng thiếu biến mục tiêu (không điền giá trị cho mục tiêu để tránh sai lệch).")
    for c in features:   # bool → int để Imputer/Scaler xử lý được
        if pd.api.types.is_bool_dtype(data[c]):
            data[c] = data[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(data[c]):
            data[c] = data[c].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    problem = problem or detect_problem_type(data[target])[0]

    if problem == "classification":
        data[target] = data[target].astype(str)
        counts = data[target].value_counts()
        if counts.size > 30:
            raise ValueError(f"Biến mục tiêu có {counts.size} lớp - quá nhiều cho phân loại. Hãy chọn hồi quy hoặc gộp lớp.")
        rare = counts[counts < max(5, CV_FOLDS)].index
        if len(rare):
            data = data[~data[target].isin(rare)]
            notes.append(f"Loại {len(rare)} lớp hiếm (< {max(5, CV_FOLDS)} mẫu) để có thể chia tầng (stratify).")
        if data[target].nunique() < 2:
            raise ValueError("Sau khi lọc, biến mục tiêu chỉ còn 1 lớp.")
    else:
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target])

    if len(data) > max_rows:
        strat = data[target] if problem == "classification" else None
        data, _ = train_test_split(data, train_size=max_rows, random_state=RANDOM_STATE, stratify=strat)
        notes.append(f"Lấy mẫu ngẫu nhiên {max_rows:,} dòng (random_state={RANDOM_STATE}) để huấn luyện trong thời gian hợp lý.")
    if len(data) < 30:
        raise ValueError(f"Chỉ có {len(data)} dòng hợp lệ - không đủ để huấn luyện và đánh giá tin cậy (cần ≥ 30).")

    X, y = data[features], data[target]
    stratify = y if problem == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify)

    balanced = False
    classes: list = []
    if problem == "classification":
        classes = sorted(y.unique().tolist())
        ratio = y_tr.value_counts(normalize=True)
        imbalance = ratio.min() / ratio.max()
        balanced = handle_imbalance == "balanced" or (handle_imbalance == "auto" and imbalance < 0.5)
        if balanced:
            notes.append(f"Mất cân bằng lớp (tỷ lệ lớp nhỏ/lớn = {imbalance:.2f}) ⇒ dùng class_weight='balanced' / sample_weight.")

    catalog = model_catalog(problem, balanced)
    names = [m for m in (model_names or list(catalog)) if m in catalog]
    if not names:
        raise ValueError("Chưa chọn mô hình hợp lệ.")

    # Baseline
    base = DummyRegressor(strategy="mean") if problem == "regression" else DummyClassifier(strategy="most_frequent")
    base_pipe = Pipeline([("prep", build_preprocessor(X_tr, False)), ("model", base)]).fit(X_tr, y_tr)
    yb = base_pipe.predict(X_te)
    base_metrics = _regression_metrics(y_te, yb) if problem == "regression" else \
        _classification_metrics(y_te, yb, None, classes)

    rows, details, pipes = [], {}, {}
    cv_n = min(len(X_tr), 8000)
    cv_idx = X_tr.sample(cv_n, random_state=RANDOM_STATE).index if cv_n < len(X_tr) else X_tr.index
    for i, name in enumerate(names):
        if progress:
            progress(i / len(names), f"Đang huấn luyện {name}...")
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor(X_tr, name in LINEAR_MODELS)), ("model", catalog[name])])
        fit_kw = {}
        if problem == "classification" and balanced and "Gradient Boosting" in name:
            fit_kw["model__sample_weight"] = compute_sample_weight("balanced", y_tr)
        pipe.fit(X_tr, y_tr, **fit_kw)
        fit_time = time.time() - t0
        pred = pipe.predict(X_te)
        pred_tr = pipe.predict(X_tr)
        if problem == "regression":
            m = _regression_metrics(y_te, pred)
            m_tr = _regression_metrics(y_tr, pred_tr)
            row = {"Mô hình": name, "MAE": m["MAE"], "MSE": m["MSE"], "RMSE": m["RMSE"], "R2": m["R2"],
                   "R2 (train)": m_tr["R2"]}
            scoring, folds = "neg_root_mean_squared_error", KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        else:
            proba = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None
            m = _classification_metrics(y_te, pred, proba, classes)
            m_tr = _classification_metrics(y_tr, pred_tr, None, classes)
            row = {"Mô hình": name, **{k: m[k] for k in ("Accuracy", "Precision", "Recall", "F1 (weighted)", "F1 (macro)", "ROC-AUC")},
                   "F1 macro (train)": m_tr["F1 (macro)"]}
            scoring, folds = "f1_macro", StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        if cv:
            try:
                scores = cross_val_score(Pipeline([("prep", build_preprocessor(X_tr, name in LINEAR_MODELS)),
                                                   ("model", catalog[name])]),
                                         X_tr.loc[cv_idx], y_tr.loc[cv_idx], cv=folds, scoring=scoring, n_jobs=None)
                scores = -scores if scoring.startswith("neg_") else scores
                row["CV mean"], row["CV std"] = float(np.mean(scores)), float(np.std(scores))
            except Exception as exc:
                row["CV mean"], row["CV std"] = np.nan, np.nan
                notes.append(f"Cross-validation lỗi với {name}: {exc}")
        row["Thời gian (s)"] = round(fit_time, 2)
        rows.append(row)
        det = {"metrics": m, "train_metrics": m_tr, "importance": _importance(pipe), "fit_time": fit_time}
        if problem == "classification":
            det["confusion"] = pd.DataFrame(confusion_matrix(y_te, pred, labels=classes), index=classes, columns=classes)
            det["y_score"] = proba[:, 1] if (proba is not None and len(classes) == 2) else None
            det["y_true"] = np.asarray(y_te)
        else:
            det["residual_sample"] = pd.DataFrame({"Thực tế": np.asarray(y_te)[:2000], "Dự đoán": np.asarray(pred)[:2000]})
        details[name] = det
        pipes[name] = pipe

    comp = pd.DataFrame(rows)
    if problem == "regression":
        best = comp.sort_values(["RMSE", "R2"], ascending=[True, False]).iloc[0]["Mô hình"]
        b = comp.set_index("Mô hình").loc[best]
        improve = (1 - b["RMSE"] / base_metrics["RMSE"]) * 100 if base_metrics["RMSE"] else np.nan
        reason = (f"RMSE trên tập test thấp nhất ({b['RMSE']:.4g}); R² = {b['R2']:.3f}. "
                  f"Giảm {improve:.1f}% RMSE so với baseline dự đoán trung bình (RMSE = {base_metrics['RMSE']:.4g}).")
        gap = b["R2 (train)"] - b["R2"]
    else:
        best = comp.sort_values(["F1 (macro)", "Accuracy"], ascending=False).iloc[0]["Mô hình"]
        b = comp.set_index("Mô hình").loc[best]
        reason = (f"F1-macro trên tập test cao nhất ({b['F1 (macro)']:.3f}) - chỉ số cân bằng giữa các lớp, phù hợp khi lớp mất cân bằng; "
                  f"Accuracy = {b['Accuracy']:.3f} so với baseline đoán lớp đa số = {base_metrics['Accuracy']:.3f}.")
        gap = b["F1 macro (train)"] - b["F1 (macro)"]
    if gap > 0.1:
        notes.append(f"Mô hình tốt nhất có chênh lệch train-test = {gap:.3f} ⇒ có dấu hiệu overfitting, cần thận trọng khi tổng quát hóa.")
    elif gap < -0.1:
        notes.append(f"Điểm trên tập test cao hơn tập train ({-gap:.3f}) - bất thường, thường do ngoại lệ/giá trị cực đoan rơi vào tập train; "
                     "nên xem xét xử lý ngoại lệ của biến mục tiêu và đối chiếu kết quả cross-validation.")
    else:
        notes.append(f"Chênh lệch train-test của mô hình tốt nhất = {gap:.3f} ⇒ khả năng tổng quát hóa ổn định.")
    if "CV std" in comp:
        sd = comp.set_index("Mô hình").loc[best, "CV std"]
        if pd.notna(sd):
            notes.append(f"Độ lệch chuẩn qua {CV_FOLDS}-fold CV của mô hình tốt nhất = {sd:.4g} (càng nhỏ càng ổn định).")
    base_row = {"Mô hình": BASELINE_NAME}
    base_row.update({k: v for k, v in base_metrics.items() if k in comp.columns})
    comp = pd.concat([comp, pd.DataFrame([base_row])], ignore_index=True)
    num_cols = comp.select_dtypes("number").columns
    comp[num_cols] = comp[num_cols].round(4)
    if progress:
        progress(1.0, "Hoàn tất")
    return TrainResult(problem, target, features, comp, best, reason, details, pipes, notes,
                       len(X_tr), len(X_te), classes, baseline=base_metrics)


def save_model(result: TrainResult, dataset_id: str, name: str | None = None) -> str:
    name = name or result.best_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() else "_" for ch in f"{result.target}_{name}")[:60]
    path = MODELS_DIR / f"{dataset_id}_{safe}_{stamp}.joblib"
    joblib.dump({"pipeline": result.pipelines[name], "target": result.target, "features": result.features,
                 "problem": result.problem, "model_name": name, "classes": result.classes,
                 "metrics": result.details[name]["metrics"], "created": stamp}, path)
    meta = path.with_suffix(".json")
    meta.write_text(json.dumps({"model": name, "target": result.target, "features": result.features,
                                "problem": result.problem, "metrics": result.details[name]["metrics"]},
                               ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    result.saved_path = str(path)
    return str(path)


def load_model(path: str) -> dict:
    return joblib.load(path)


def predict_one(pipeline: Pipeline, features: list[str], values: dict):
    row = pd.DataFrame([{f: values.get(f, np.nan) for f in features}])
    pred = pipeline.predict(row)[0]
    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = dict(zip(pipeline.classes_, pipeline.predict_proba(row)[0].round(4)))
        except Exception:
            proba = None
    return pred, proba
