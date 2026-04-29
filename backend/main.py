from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import io, math, json, asyncio, base64, sqlite3, threading
from datetime import datetime
from pathlib import Path
from collections import Counter

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
    r2_score, silhouette_score, roc_auc_score,
    davies_bouldin_score, calinski_harabasz_score,
    classification_report, explained_variance_score,
    max_error, median_absolute_error
)
from pydantic import BaseModel, ConfigDict
from typing import List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# SQLite — history / recent works
# ─────────────────────────────────────────────
DB_PATH  = Path("ml_platform.db")
_db_lock = threading.Lock()

def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                filename    TEXT,
                n_rows      INTEGER,
                n_cols      INTEGER,
                model_type  TEXT,
                model_name  TEXT,
                metric_key  TEXT,
                metric_val  REAL,
                status      TEXT    DEFAULT 'completed',
                summary     TEXT
            )
        """)
        conn.commit()

init_db()

def db_save(data: dict):
    with _db_lock:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO sessions
                    (created_at,filename,n_rows,n_cols,model_type,model_name,
                     metric_key,metric_val,status,summary)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.utcnow().isoformat(),
                data.get("filename","unknown"),
                data.get("n_rows"), data.get("n_cols"),
                data.get("model_type"), data.get("model_name"),
                data.get("metric_key"), data.get("metric_val"),
                data.get("status","completed"), data.get("summary"),
            ))
            conn.commit()

def db_recent(limit=20):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

def db_delete(sid: int):
    with _db_lock:
        with get_db() as conn:
            conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
            conn.commit()

def db_clear():
    with _db_lock:
        with get_db() as conn:
            conn.execute("DELETE FROM sessions")
            conn.commit()

# ─────────────────────────────────────────────
# HISTORY ENDPOINTS
# ─────────────────────────────────────────────
@app.get('/history')
async def get_history(limit: int = 20):
    return {"sessions": db_recent(limit)}

@app.delete('/history/{session_id}')
async def delete_session(session_id: int):
    db_delete(session_id)
    return {"ok": True}

@app.delete('/history')
async def clear_history():
    db_clear()
    return {"ok": True}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def make_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  RobustScaler()),
        ('model',   model),
    ])

def overfitting_status(train_score, test_score, threshold=0.12):
    gap = train_score - test_score
    if gap > threshold:
        return 'overfitting', (
            f"Overfitting: train {train_score:.1%} vs test {test_score:.1%}. "
            "Try a simpler model or more data."
        )
    if test_score < 0.45 and train_score < 0.5:
        return 'underfitting', (
            "Underfitting: model too simple. Try Random Forest or add features."
        )
    return 'good', None

def decode_csv(csv_data: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(base64.b64decode(csv_data).decode('utf-8')))

def safe_stratify(y_arr):
    """Return y_arr for stratify only when every class has ≥ 2 samples."""
    counts = Counter(y_arr.tolist() if hasattr(y_arr, 'tolist') else list(y_arr))
    return y_arr if min(counts.values()) >= 2 else None

def safe_cv(pipe, X, y, cv=5, scoring='accuracy'):
    """Cross-validate, automatically reducing folds if classes are too small."""
    counts = Counter(y.tolist() if hasattr(y, 'tolist') else list(y))
    min_count = min(counts.values())
    n_splits  = min(cv, min_count, len(X) // max(len(counts), 1))
    n_splits  = max(2, n_splits)
    try:
        scores = cross_val_score(pipe, X, y, cv=n_splits, scoring=scoring, n_jobs=-1)
        return round(float(scores.mean()), 4), round(float(scores.std()), 4), n_splits
    except Exception:
        return None, None, n_splits

def safe_report(y_test, y_pred, le):
    """
    Build classification_report safely — use only the labels that actually
    appear in y_test (not all fitted classes), so target_names length matches.
    """
    present = sorted(set(y_test))
    names   = [str(le.classes_[i]) for i in present]
    return classification_report(
        y_test, y_pred,
        labels=present,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if len(df) < 10:
        raise HTTPException(400, "Dataset must have at least 10 rows.")
    return {
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "rows": len(df),
    }

# ─────────────────────────────────────────────
# ANALYSE  (visualisation page)
# ─────────────────────────────────────────────
@app.post('/analyse')
async def analyse(file: UploadFile = File(...)):
    contents = await file.read()
    df     = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    num_df = df.select_dtypes(include=[np.number])

    histograms = {}
    for col in num_df.columns:
        counts, bin_edges = np.histogram(num_df[col].dropna(), bins=20)
        histograms[col] = {'counts': counts.tolist(),
                           'bins': [round(float(b), 4) for b in bin_edges[:-1]]}

    corr = num_df.corr().round(2)
    missing = [{'column': c,
                'missing': int(df[c].isnull().sum()),
                'pct': round(df[c].isnull().sum() / len(df) * 100, 1)}
               for c in df.columns]

    boxplots = {}
    for col in num_df.columns:
        s  = num_df[col].dropna()
        q1 = float(s.quantile(0.25)); q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        boxplots[col] = {
            'min': round(float(s.min()), 4), 'q1': round(q1, 4),
            'median': round(float(s.median()), 4),
            'q3': round(q3, 4), 'max': round(float(s.max()), 4),
            'outliers': [round(float(v), 4) for v in
                         s[(s < q1-1.5*iqr) | (s > q3+1.5*iqr)].head(50)]
        }

    return {
        'numeric_columns': list(num_df.columns),
        'histograms': histograms,
        'correlation': {'columns': list(corr.columns),
                        'matrix': corr.fillna(0).values.tolist()},
        'scatter_data': num_df.head(200).fillna(0).to_dict(orient='records'),
        'missing': missing,
        'boxplots': boxplots,
    }

# ─────────────────────────────────────────────
# DEEP DATASET ANALYSIS + MODEL SUGGESTION
# ─────────────────────────────────────────────
class AnalyseDatasetRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    csv_data: str
    filename: Optional[str] = "dataset.csv"

@app.post('/analyse-dataset')
async def analyse_dataset(req: AnalyseDatasetRequest):
    df     = decode_csv(req.csv_data)
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(exclude=[np.number])

    n_rows, n_cols = len(df), len(df.columns)
    n_num, n_cat   = len(num_df.columns), len(cat_df.columns)

    missing_pct       = df.isnull().sum() / len(df) * 100
    cols_high_missing = missing_pct[missing_pct > 30].index.tolist()
    total_missing_pct = round(float(df.isnull().values.mean() * 100), 2)
    n_dupes           = int(df.duplicated().sum())

    outlier_cols = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0: continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
        if n_out > 0:
            outlier_cols.append({"column": col, "count": n_out,
                                 "pct": round(n_out/len(s)*100, 1)})

    high_corr_pairs = []
    if n_num >= 2:
        cm = num_df.corr().abs()
        cols = cm.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                v = cm.iloc[i, j]
                if pd.notna(v) and v > 0.85:
                    high_corr_pairs.append({"col_a": cols[i], "col_b": cols[j],
                                            "r": round(float(v), 3)})

    skewed_cols = [{"column": c, "skew": round(float(num_df[c].skew()), 2)}
                   for c in num_df.columns if abs(float(num_df[c].skew())) > 1]

    cat_info = [{"column": c, "unique": int(df[c].nunique()),
                 "pct": round(df[c].nunique()/len(df)*100, 1)}
                for c in cat_df.columns]

    # Classification candidates
    target_suggestions = []
    for col in cat_df.columns:
        vc = df[col].value_counts()
        if 2 <= len(vc) <= 30:
            min_pct = round(float(vc.min()/len(df)*100), 1)
            target_suggestions.append({
                "column": col, "n_classes": int(len(vc)),
                "min_class_pct": min_pct, "imbalanced": min_pct < 10,
                "classes": vc.index.tolist()[:8],
            })
    # Numeric low-cardinality as classification
    for col in num_df.columns:
        vc = df[col].value_counts()
        if 2 <= len(vc) <= 10:
            min_pct = round(float(vc.min()/len(df)*100), 1)
            target_suggestions.append({
                "column": col, "n_classes": int(len(vc)),
                "min_class_pct": min_pct, "imbalanced": min_pct < 10,
                "classes": [str(v) for v in vc.index.tolist()[:8]],
                "is_numeric": True,
            })

    # Build suggestions
    suggestions = []

    # Classification
    if target_suggestions:
        best = target_suggestions[0]
        reasons = [
            f"Found {best['n_classes']} classes in '{best['column']}'",
            f"{n_num} numeric feature(s) available",
        ]
        if n_rows < 500:       rec = "Logistic Regression or Decision Tree"; reasons.append("small dataset — simpler models generalise better")
        elif best['imbalanced']: rec = "Random Forest (class_weight='balanced')"; reasons.append("class imbalance detected")
        elif n_num > 10:        rec = "Random Forest or SVM"; reasons.append("many features — ensemble or margin models work well")
        else:                   rec = "Random Forest"; reasons.append("good dataset size for ensembles")
        suggestions.append({
            "type": "classification", "label": "Classification",
            "icon": "◈", "accent": "#6c63ff",
            "confidence": "high" if n_num >= 2 else "medium",
            "reason": ". ".join(reasons) + ".",
            "recommended_model": rec,
            "target_col": best['column'], "n_classes": best['n_classes'],
        })

    # Regression
    reg_col = None
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0: continue
        cv_coef  = float(s.std() / (abs(s.mean()) + 1e-8))
        n_unique = s.nunique()
        if cv_coef > 0.1 and n_unique > 10:
            reg_col = col; break
    if reg_col:
        reasons = [f"'{reg_col}' is continuous — ideal regression target",
                   f"{n_num-1} other numeric feature(s) available"]
        rec = "Random Forest Regressor" if n_rows > 5000 else "Ridge or Random Forest"
        suggestions.append({
            "type": "regression", "label": "Regression",
            "icon": "◉", "accent": "#0ea5e9",
            "confidence": "high" if n_num >= 3 else "medium",
            "reason": ". ".join(reasons) + ".",
            "recommended_model": rec, "target_col": reg_col,
        })

    # Clustering
    suggestions.append({
        "type": "clustering", "label": "Clustering",
        "icon": "◎", "accent": "#10b981",
        "confidence": "medium" if not target_suggestions else "low",
        "reason": (f"No categorical target — clustering finds natural groups across {n_num} features."
                   if not target_suggestions
                   else f"Clustering can reveal hidden structure across {n_num} numeric features."),
        "recommended_model": "K-Means (fast)" if n_rows > 10000 else "K-Means or DBSCAN",
    })

    # Neural network
    if n_rows >= 500 and n_num >= 3:
        suggestions.append({
            "type": "neural-network", "label": "Neural Network",
            "icon": "◌", "accent": "#f59e0b",
            "confidence": "medium" if n_rows >= 2000 else "low",
            "reason": f"MLP models complex patterns across {n_num} features × {n_rows:,} rows.",
            "recommended_model": "MLP 64→32 ReLU, early stopping",
        })

    # Preprocessing plan
    preprocess_steps = []
    if total_missing_pct > 0:
        preprocess_steps.append({
            "step": "Impute missing values",
            "detail": f"{total_missing_pct}% missing — numeric→median, categorical→mode",
            "severity": "high" if total_missing_pct > 10 else "medium", "auto": True,
        })
    if cols_high_missing:
        preprocess_steps.append({
            "step": "Drop high-missing columns",
            "detail": f"{', '.join(cols_high_missing[:3])} have >30% missing — will be dropped",
            "severity": "high", "auto": True,
        })
    if n_dupes > 0:
        preprocess_steps.append({
            "step": "Remove duplicates",
            "detail": f"{n_dupes} exact duplicates ({round(n_dupes/n_rows*100,1)}%) — will be removed",
            "severity": "medium", "auto": True,
        })
    if outlier_cols:
        preprocess_steps.append({
            "step": "Outlier handling",
            "detail": f"{outlier_cols[0]['column']} has {outlier_cols[0]['count']} outliers — RobustScaler reduces impact",
            "severity": "medium", "auto": True,
        })
    if skewed_cols:
        preprocess_steps.append({
            "step": "Skewed distributions",
            "detail": f"{len(skewed_cols)} column(s) highly skewed — RobustScaler applied",
            "severity": "low", "auto": True,
        })
    if high_corr_pairs:
        preprocess_steps.append({
            "step": "Highly correlated features",
            "detail": f"{len(high_corr_pairs)} pair(s) with r>0.85 — consider removing one per pair",
            "severity": "low", "auto": False,
        })
    if not preprocess_steps:
        preprocess_steps.append({
            "step": "Dataset looks clean ✓",
            "detail": "No significant issues. Standard pipeline will work well.",
            "severity": "ok", "auto": True,
        })
    preprocess_steps.append({
        "step": "Feature scaling",
        "detail": "RobustScaler applied to all numeric features automatically",
        "severity": "ok", "auto": True,
    })

    health = 100
    health -= min(40, total_missing_pct * 2)
    health -= min(10, n_dupes / max(n_rows, 1) * 100)
    health -= len(high_corr_pairs) * 3
    health -= len(skewed_cols) * 2
    health  = max(0, round(health))

    db_save({
        "filename": req.filename, "n_rows": n_rows, "n_cols": n_cols,
        "model_type": suggestions[0]["type"] if suggestions else "unknown",
        "model_name": "auto-detected", "metric_key": "health",
        "metric_val": health, "status": "analysed",
        "summary": f"{n_rows} rows, {n_cols} cols, health {health}/100",
    })

    return {
        "n_rows": n_rows, "n_cols": n_cols,
        "n_numeric": n_num, "n_categorical": n_cat,
        "total_missing_pct": total_missing_pct, "n_duplicates": n_dupes,
        "outlier_cols": outlier_cols[:5], "high_corr_pairs": high_corr_pairs[:5],
        "skewed_cols": skewed_cols[:5], "cat_info": cat_info,
        "target_suggestions": target_suggestions, "suggestions": suggestions,
        "preprocess_steps": preprocess_steps, "health_score": health,
    }

# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────
class PreprocessRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    csv_data: str
    drop_duplicates: Optional[bool] = True
    drop_high_missing: Optional[bool] = True
    high_missing_threshold: Optional[float] = 0.3

@app.post('/preprocess')
async def preprocess(req: PreprocessRequest):
    df = decode_csv(req.csv_data)
    steps, orig = [], df.shape

    if req.drop_duplicates:
        n_before = len(df); df = df.drop_duplicates()
        if (r := n_before - len(df)) > 0:
            steps.append(f"Removed {r} duplicate rows")

    if req.drop_high_missing:
        miss = df.isnull().sum() / len(df)
        drop = miss[miss > req.high_missing_threshold].index.tolist()
        if drop:
            df = df.drop(columns=drop)
            steps.append(f"Dropped {len(drop)} high-missing column(s): {', '.join(drop[:3])}")

    total_miss = int(df.isnull().sum().sum())
    if total_miss > 0:
        for c in df.select_dtypes(include=[np.number]).columns:
            if df[c].isnull().any(): df[c] = df[c].fillna(df[c].median())
        for c in df.select_dtypes(exclude=[np.number]).columns:
            if df[c].isnull().any():
                mode = df[c].mode()
                df[c] = df[c].fillna(mode[0] if len(mode) > 0 else 'Unknown')
        steps.append(f"Imputed {total_miss} missing values (numeric→median, categorical→mode)")

    buf = io.StringIO(); df.to_csv(buf, index=False)
    clean_b64 = base64.b64encode(buf.getvalue().encode()).decode()

    return {
        "original_rows": orig[0], "original_cols": orig[1],
        "final_rows": len(df), "final_cols": len(df.columns),
        "rows_removed": orig[0] - len(df),
        "cols_removed": orig[1] - len(df.columns),
        "remaining_missing": int(df.isnull().sum().sum()),
        "steps_applied": steps, "clean_csv_data": clean_b64,
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
    }

# ─────────────────────────────────────────────
# CLASSIFICATION  ── all bugs fixed
# ─────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    target: str
    features: List[str]
    test_size: float
    csv_data: str
    filename: Optional[str] = "dataset.csv"

@app.post('/classify')
async def classify(req: ClassifyRequest):
    async def stream():
        def emit(stage, pct, msg, data=None):
            p = {"stage": stage, "pct": pct, "msg": msg}
            if data: p["data"] = data
            return f"data: {json.dumps(p)}\n\n"

        yield emit("preprocess", 5, "Decoding dataset...")
        await asyncio.sleep(0.05)

        df    = decode_csv(req.csv_data)
        avail = [f for f in req.features if f in df.columns]
        X     = df[avail].select_dtypes(include=[np.number])
        y_raw = df[req.target]

        if X.shape[1] == 0:
            yield emit("error", 0, "No numeric feature columns found."); return
        if len(X) < 20:
            yield emit("error", 0, "Need at least 20 rows to train."); return

        total_miss = int(X.isnull().sum().sum())
        yield emit("preprocess", 15, f"{total_miss} missing values → filling with median...")
        await asyncio.sleep(0.1)

        le    = LabelEncoder()
        y_enc = le.fit_transform(y_raw.astype(str))
        n_cls = len(le.classes_)

        yield emit("preprocess", 28,
                   f"Encoded {n_cls} classes. "
                   f"Splitting {int((1-req.test_size)*100)}/{int(req.test_size*100)}%...")
        await asyncio.sleep(0.1)

        # ── FIX: safe stratify ──
        strat = safe_stratify(y_enc)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=req.test_size, random_state=42, stratify=strat
        )

        yield emit("preprocess", 42,
                   f"RobustScaler applied. {len(X_train)} train / {len(X_test)} test rows.")
        await asyncio.sleep(0.15)

        # ── FIX: use 'is None' not falsy check ──
        base_models = {
            'logistic_regression': LogisticRegression(
                max_iter=2000, C=1.0, class_weight='balanced', solver='lbfgs'),
            'decision_tree': DecisionTreeClassifier(
                random_state=42, max_depth=6, min_samples_leaf=3, class_weight='balanced'),
            'random_forest': RandomForestClassifier(
                n_estimators=200, random_state=42, max_depth=10,
                min_samples_leaf=2, class_weight='balanced', n_jobs=-1),
            'svm': SVC(probability=True, random_state=42, kernel='rbf',
                       C=1.0, class_weight='balanced'),
            'knn': KNeighborsClassifier(
                n_neighbors=min(7, max(1, len(X_train)//10)), weights='distance'),
        }
        base = base_models.get(req.model_name)
        if base is None:
            yield emit("error", 0, f"Unknown model: {req.model_name}"); return

        yield emit("train", 50,
                   f"Training {req.model_name.replace('_',' ').title()} "
                   f"on {len(X_train)} samples...")
        await asyncio.sleep(0.2)

        pipe = make_pipeline(base)
        pipe.fit(X_train, y_train)

        yield emit("train", 67, "Fitted. Running predictions on test set...")
        await asyncio.sleep(0.1)

        y_pred       = pipe.predict(X_test)
        y_pred_train = pipe.predict(X_train)
        y_prob = None
        try: y_prob = pipe.predict_proba(X_test)
        except Exception: pass

        yield emit("evaluate", 75, "Computing classification metrics...")
        await asyncio.sleep(0.1)

        avg       = 'weighted' if n_cls > 2 else 'binary'
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test,  y_pred)
        fit_status, fit_msg = overfitting_status(train_acc, test_acc)

        roc_auc = None
        try:
            if y_prob is not None:
                if n_cls == 2:
                    roc_auc = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
                else:
                    roc_auc = round(float(roc_auc_score(
                        y_test, y_prob, multi_class='ovr', average='weighted')), 4)
        except Exception: pass

        yield emit("evaluate", 83, "Running cross-validation...")
        await asyncio.sleep(0.1)

        # ── FIX: safe_cv handles tiny classes ──
        cv_mean_raw, cv_std_raw, n_splits = safe_cv(pipe, X, y_enc, cv=5, scoring='accuracy')
        cv_mean = round(cv_mean_raw * 100, 2) if cv_mean_raw is not None else None
        cv_std  = round(cv_std_raw  * 100, 2) if cv_std_raw  is not None else None

        yield emit("evaluate", 92, "Computing feature importance...")
        await asyncio.sleep(0.05)

        mo       = pipe.named_steps['model']
        feat_imp = []
        if hasattr(mo, 'feature_importances_'):
            feat_imp = sorted(
                [{"feature": f, "importance": round(float(i), 4)}
                 for f, i in zip(X.columns, mo.feature_importances_)],
                key=lambda x: x["importance"], reverse=True
            )
        elif hasattr(mo, 'coef_'):
            coefs = np.abs(mo.coef_[0]) if mo.coef_.ndim > 1 else np.abs(mo.coef_)
            feat_imp = sorted(
                [{"feature": f, "importance": round(float(i), 4)}
                 for f, i in zip(X.columns, coefs)],
                key=lambda x: x["importance"], reverse=True
            )

        # ── FIX: safe_report matches labels to what's in y_test ──
        report = safe_report(y_test, y_pred, le)

        # Labels for confusion matrix — only those present in y_test
        test_labels_idx = sorted(set(y_test))
        cm_labels = [str(le.classes_[i]) for i in test_labels_idx]

        pre_stats = {
            "total_rows": len(df), "train_rows": len(X_train),
            "test_rows": len(X_test), "n_features": X.shape[1],
            "missing_filled": total_miss, "scaler": "RobustScaler",
            "label_encoded": True, "n_classes": n_cls,
            "class_names": le.classes_.tolist(),
        }

        db_save({
            "filename": req.filename, "n_rows": len(df), "n_cols": len(df.columns),
            "model_type": "classification", "model_name": req.model_name,
            "metric_key": "accuracy", "metric_val": round(test_acc * 100, 2),
            "summary": (f"{req.model_name.replace('_',' ').title()} · "
                        f"Acc {round(test_acc*100,2)}%"
                        + (f" · CV {cv_mean}%" if cv_mean else "")),
        })

        yield emit("done", 100, "Training complete!", {
            "accuracy":          round(test_acc * 100, 2),
            "train_accuracy":    round(train_acc * 100, 2),
            "f1":         round(f1_score(y_test, y_pred, average=avg, zero_division=0)*100, 2),
            "precision":  round(precision_score(y_test, y_pred, average=avg, zero_division=0)*100, 2),
            "recall":     round(recall_score(y_test, y_pred, average=avg, zero_division=0)*100, 2),
            "roc_auc":           roc_auc,
            "cv_accuracy":       cv_mean,
            "cv_std":            cv_std,
            "fit_status":        fit_status,
            "fit_warning":       fit_msg,
            "confusion_matrix":  confusion_matrix(y_test, y_pred, labels=test_labels_idx).tolist(),
            "labels":            cm_labels,
            "feature_importance": feat_imp,
            "class_report":      report,
            "preprocess_stats":  pre_stats,
            "n_train":           len(X_train),
            "n_test":            len(X_test),
            "n_classes":         n_cls,
        })

    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ─────────────────────────────────────────────
# REGRESSION
# ─────────────────────────────────────────────
class RegressRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    target: str
    features: List[str]
    test_size: float
    csv_data: str
    filename: Optional[str] = "dataset.csv"

@app.post('/regress')
async def regress(req: RegressRequest):
    async def stream():
        def emit(stage, pct, msg, data=None):
            p = {"stage": stage, "pct": pct, "msg": msg}
            if data: p["data"] = data
            return f"data: {json.dumps(p)}\n\n"

        yield emit("preprocess", 5, "Decoding dataset...")
        await asyncio.sleep(0.05)

        df    = decode_csv(req.csv_data)
        avail = [f for f in req.features if f in df.columns]
        X     = df[avail].select_dtypes(include=[np.number])
        y     = pd.to_numeric(df[req.target], errors='coerce')

        if X.shape[1] == 0:
            yield emit("error", 0, "No numeric feature columns found."); return
        if y.isnull().all():
            yield emit("error", 0, "Target column has no numeric values."); return

        valid = ~y.isnull(); X, y = X[valid], y[valid]
        total_miss = int(X.isnull().sum().sum())

        yield emit("preprocess", 18,
                   f"Filled {total_miss} missing. "
                   f"Target: {y.min():.2f}–{y.max():.2f}")
        await asyncio.sleep(0.1)
        yield emit("preprocess", 32, f"Splitting {int((1-req.test_size)*100)}/{int(req.test_size*100)}%...")
        await asyncio.sleep(0.1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=req.test_size, random_state=42)

        yield emit("preprocess", 44, "Applying RobustScaler...")
        await asyncio.sleep(0.1)

        base_models = {
            'linear_regression': LinearRegression(),
            'ridge':             Ridge(alpha=1.0),
            'lasso':             Lasso(alpha=0.1, max_iter=5000),
            'decision_tree':     DecisionTreeRegressor(random_state=42, max_depth=6, min_samples_leaf=3),
            'random_forest':     RandomForestRegressor(n_estimators=200, random_state=42,
                                                        max_depth=10, min_samples_leaf=2, n_jobs=-1),
        }
        base = base_models.get(req.model_name)
        if base is None:
            yield emit("error", 0, f"Unknown model: {req.model_name}"); return

        yield emit("train", 52,
                   f"Training {req.model_name.replace('_',' ').title()} on {len(X_train)} samples...")
        await asyncio.sleep(0.2)

        pipe = make_pipeline(base)
        pipe.fit(X_train, y_train)

        yield emit("train", 67, "Fitted. Generating predictions...")
        await asyncio.sleep(0.1)

        y_pred       = pipe.predict(X_test)
        y_pred_train = pipe.predict(X_train)

        yield emit("evaluate", 75, "Computing regression metrics...")
        await asyncio.sleep(0.1)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2  = r2_score(y_test,  y_pred)
        mse      = mean_squared_error(y_test, y_pred)
        fit_status, fit_msg = overfitting_status(max(0, train_r2), max(0, test_r2))

        mape    = float(np.mean(np.abs((y_test-y_pred)/(np.abs(y_test)+1e-8)))*100)
        exp_var = float(explained_variance_score(y_test, y_pred))
        max_err = float(max_error(y_test, y_pred))
        med_ae  = float(median_absolute_error(y_test, y_pred))
        n_f     = X_test.shape[1]
        adj_r2  = 1-(1-test_r2)*(len(y_test)-1)/(len(y_test)-n_f-1) if n_f>0 else test_r2

        yield emit("evaluate", 83, "Running cross-validation...")
        await asyncio.sleep(0.1)

        cv_mean_r, cv_std_r, _ = safe_cv(pipe, X, y, cv=5, scoring='r2')
        cv_mean = round(cv_mean_r, 4) if cv_mean_r is not None else None
        cv_std  = round(cv_std_r,  4) if cv_std_r  is not None else None

        yield emit("evaluate", 92, "Computing feature importance...")
        await asyncio.sleep(0.05)

        mo = pipe.named_steps['model']
        feat_imp = []
        if hasattr(mo, 'feature_importances_'):
            feat_imp = sorted(
                [{"feature": f, "importance": round(float(i), 4)}
                 for f, i in zip(X.columns, mo.feature_importances_)],
                key=lambda x: x["importance"], reverse=True
            )
        elif hasattr(mo, 'coef_'):
            feat_imp = sorted(
                [{"feature": f, "importance": round(float(abs(i)), 4)}
                 for f, i in zip(X.columns, mo.coef_)],
                key=lambda x: x["importance"], reverse=True
            )

        scatter   = [{"actual": round(float(a),4), "predicted": round(float(p),4)}
                     for a,p in zip(y_test[:200], y_pred[:200])]
        residuals = [round(float(a-p),4) for a,p in zip(y_test[:200], y_pred[:200])]

        db_save({
            "filename": req.filename, "n_rows": len(df), "n_cols": len(df.columns),
            "model_type": "regression", "model_name": req.model_name,
            "metric_key": "r2", "metric_val": round(test_r2, 4),
            "summary": (f"{req.model_name.replace('_',' ').title()} · "
                        f"R² {round(test_r2,4)}"
                        + (f" · CV {cv_mean}" if cv_mean else "")),
        })

        yield emit("done", 100, "Training complete!", {
            "mae":          round(float(mean_absolute_error(y_test,y_pred)), 4),
            "mse":          round(float(mse), 4),
            "rmse":         round(float(math.sqrt(mse)), 4),
            "r2":           round(float(test_r2), 4),
            "adj_r2":       round(float(adj_r2), 4),
            "train_r2":     round(float(train_r2), 4),
            "mape":         round(mape, 2),
            "explained_var":round(exp_var, 4),
            "max_error":    round(max_err, 4),
            "median_ae":    round(med_ae, 4),
            "cv_r2":        cv_mean,
            "cv_std":       cv_std,
            "fit_status":   fit_status,
            "fit_warning":  fit_msg,
            "feature_importance": feat_imp,
            "scatter":      scatter,
            "residuals":    residuals,
            "preprocess_stats": {
                "total_rows": len(df), "train_rows": len(X_train),
                "test_rows": len(X_test), "n_features": X.shape[1],
                "missing_filled": total_miss, "scaler": "RobustScaler",
                "target_mean": round(float(y.mean()),4),
                "target_std":  round(float(y.std()), 4),
            },
            "n_train": len(X_train), "n_test": len(X_test),
        })

    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────
class ClusterRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    features: List[str]
    csv_data: str
    n_clusters: Optional[int] = 3
    eps: Optional[float] = 0.5
    min_samples: Optional[int] = 5
    filename: Optional[str] = "dataset.csv"

@app.post('/cluster')
async def cluster(req: ClusterRequest):
    async def stream():
        def emit(stage, pct, msg, data=None):
            p = {"stage": stage, "pct": pct, "msg": msg}
            if data: p["data"] = data
            return f"data: {json.dumps(p)}\n\n"

        yield emit("preprocess", 8, "Decoding dataset...")
        await asyncio.sleep(0.08)

        df    = decode_csv(req.csv_data)
        avail = [f for f in req.features if f in df.columns]
        X_raw = df[avail].select_dtypes(include=[np.number])

        total_miss = int(X_raw.isnull().sum().sum())
        yield emit("preprocess", 22, f"Imputing {total_miss} missing values...")
        await asyncio.sleep(0.1)

        X_imp    = SimpleImputer(strategy='median').fit_transform(X_raw)
        X_scaled = RobustScaler().fit_transform(X_imp)

        yield emit("preprocess", 38, "RobustScaler applied.")
        await asyncio.sleep(0.1)

        if req.model_name == 'kmeans':
            model = KMeans(n_clusters=req.n_clusters, random_state=42, n_init='auto', max_iter=500)
        elif req.model_name == 'dbscan':
            model = DBSCAN(eps=req.eps, min_samples=req.min_samples)
        elif req.model_name == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=req.n_clusters)
        else:
            yield emit("error", 0, f"Unknown model: {req.model_name}"); return

        yield emit("train", 52,
                   f"Running {req.model_name.upper()} on "
                   f"{X_scaled.shape[0]}×{X_scaled.shape[1]}...")
        await asyncio.sleep(0.2)

        labels       = model.fit_predict(X_scaled).tolist()
        valid_labels = [l for l in set(labels) if l != -1]

        yield emit("evaluate", 65, f"Found {len(valid_labels)} clusters. Computing metrics...")
        await asyncio.sleep(0.1)

        silhouette = db_score = ch_score = inertia = None
        if len(valid_labels) >= 2:
            mask = np.array(labels) != -1
            if mask.sum() > len(valid_labels):
                try:
                    Xv = X_scaled[mask]; lv = np.array(labels)[mask]
                    silhouette = round(float(silhouette_score(Xv, lv)), 4)
                    db_score   = round(float(davies_bouldin_score(Xv, lv)), 4)
                    ch_score   = round(float(calinski_harabasz_score(Xv, lv)), 1)
                except Exception: pass

        if req.model_name == 'kmeans' and hasattr(model, 'inertia_'):
            inertia = round(float(model.inertia_), 2)

        yield emit("evaluate", 80, "PCA reduction to 2D...")
        await asyncio.sleep(0.1)

        pca    = PCA(n_components=min(2, X_scaled.shape[1]))
        coords = pca.fit_transform(X_scaled)
        pca_var = [round(float(v)*100, 1) for v in pca.explained_variance_ratio_]

        scatter = [{"x": round(float(coords[i,0]),4),
                    "y": round(float(coords[i,1]),4),
                    "cluster": labels[i]}
                   for i in range(len(labels))]

        counts = Counter(labels)
        cluster_sizes = [
            {"cluster": int(k), "size": v,
             "label": "Noise" if k==-1 else f"Cluster {k}",
             "pct": round(v/len(labels)*100, 1)}
            for k, v in sorted(counts.items())
        ]

        feat_names = list(X_raw.columns[:X_imp.shape[1]])
        X_df = pd.DataFrame(X_imp, columns=feat_names); X_df['cluster'] = labels
        cluster_profiles = []
        for c in sorted(valid_labels):
            grp = X_df[X_df['cluster']==c]
            prof = {"cluster": int(c), "label": f"Cluster {c}", "size": int(len(grp))}
            for col in feat_names: prof[col] = round(float(grp[col].mean()), 3)
            cluster_profiles.append(prof)

        db_save({
            "filename": req.filename, "n_rows": len(df), "n_cols": len(df.columns),
            "model_type": "clustering", "model_name": req.model_name,
            "metric_key": "silhouette", "metric_val": silhouette,
            "summary": (f"{req.model_name.upper()} · {len(valid_labels)} clusters"
                        + (f" · Silhouette {silhouette}" if silhouette else "")),
        })

        yield emit("done", 100, "Clustering complete!", {
            "silhouette": silhouette, "davies_bouldin": db_score,
            "calinski_harabasz": ch_score, "inertia": inertia,
            "scatter": scatter, "cluster_sizes": cluster_sizes,
            "cluster_profiles": cluster_profiles,
            "n_clusters_found": len(valid_labels),
            "n_noise": labels.count(-1),
            "pca_variance": pca_var, "total_points": len(labels),
            "preprocess_stats": {
                "total_rows": len(df), "n_features": X_raw.shape[1],
                "missing_filled": total_miss, "scaler": "RobustScaler",
                "feature_names": feat_names,
            },
        })

    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ─────────────────────────────────────────────
# NEURAL NETWORK
# ─────────────────────────────────────────────
class NeuralRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    problem_type: str
    target: str
    features: List[str]
    test_size: float
    csv_data: str
    hidden_layers: List[int]
    activation: str
    max_iter: int
    filename: Optional[str] = "dataset.csv"

@app.post('/neural')
async def neural(req: NeuralRequest):
    async def stream():
        def emit(stage, pct, msg, data=None):
            p = {"stage": stage, "pct": pct, "msg": msg}
            if data: p["data"] = data
            return f"data: {json.dumps(p)}\n\n"

        yield emit("preprocess", 5, "Decoding dataset...")
        await asyncio.sleep(0.05)

        df     = decode_csv(req.csv_data)
        avail  = [f for f in req.features if f in df.columns]
        X      = df[avail].select_dtypes(include=[np.number])
        hidden = tuple(req.hidden_layers) if req.hidden_layers else (100,)

        total_miss = int(X.isnull().sum().sum())
        yield emit("preprocess", 18, f"Imputing {total_miss} missing values...")
        await asyncio.sleep(0.1)

        X_imp    = SimpleImputer(strategy='median').fit_transform(X)
        X_scaled = StandardScaler().fit_transform(X_imp)

        yield emit("preprocess", 32, "StandardScaler applied.")
        await asyncio.sleep(0.1)

        arch = ' → '.join(str(n) for n in req.hidden_layers)

        if req.problem_type == 'classification':
            y     = df[req.target]
            le    = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            n_cls = len(le.classes_)

            # ── FIX: safe stratify ──
            strat = safe_stratify(y_enc)
            yield emit("preprocess", 42,
                       f"Split {int((1-req.test_size)*100)}/{int(req.test_size*100)}%, {n_cls} classes...")
            await asyncio.sleep(0.1)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_enc, test_size=req.test_size, random_state=42, stratify=strat
            )

            yield emit("train", 50, f"Training MLP [{arch}] · {req.activation} · max {req.max_iter} iters...")
            await asyncio.sleep(0.3)

            mdl = MLPClassifier(
                hidden_layer_sizes=hidden, activation=req.activation,
                max_iter=req.max_iter, random_state=42,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=15, learning_rate='adaptive', alpha=0.001,
            )
            mdl.fit(X_train, y_train)

            yield emit("train", 70, f"Converged in {mdl.n_iter_} iters. Evaluating...")
            await asyncio.sleep(0.1)

            y_pred       = mdl.predict(X_test)
            y_pred_train = mdl.predict(X_train)
            y_prob       = mdl.predict_proba(X_test)

            avg       = 'weighted' if n_cls > 2 else 'binary'
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc  = accuracy_score(y_test,  y_pred)
            fit_status, fit_msg = overfitting_status(train_acc, test_acc)

            roc_auc = None
            try:
                roc_auc = round(float(roc_auc_score(
                    y_test,
                    y_prob if n_cls > 2 else y_prob[:, 1],
                    multi_class='ovr' if n_cls > 2 else 'raise',
                    average='weighted' if n_cls > 2 else None
                )), 4)
            except Exception: pass

            yield emit("evaluate", 85, "Computing metrics + class report...")
            await asyncio.sleep(0.1)

            # ── FIX: safe_report ──
            report     = safe_report(y_test, y_pred, le)
            test_labels_idx = sorted(set(y_test))
            cm_labels   = [str(le.classes_[i]) for i in test_labels_idx]
            loss_curve  = [round(float(v), 6) for v in mdl.loss_curve_]
            val_curve   = [round(float(v), 6) for v in (mdl.validation_scores_ or [])]

            db_save({
                "filename": req.filename, "n_rows": len(df), "n_cols": len(df.columns),
                "model_type": "neural-classification", "model_name": f"MLP {arch}",
                "metric_key": "accuracy", "metric_val": round(test_acc*100, 2),
                "summary": f"MLP [{arch}] · Acc {round(test_acc*100,2)}% · {mdl.n_iter_} iters",
            })

            yield emit("done", 100, "Training complete!", {
                "problem_type":    "classification",
                "accuracy":        round(test_acc * 100, 2),
                "train_accuracy":  round(train_acc * 100, 2),
                "f1":       round(f1_score(y_test, y_pred, average=avg, zero_division=0)*100, 2),
                "precision":round(precision_score(y_test, y_pred, average=avg, zero_division=0)*100, 2),
                "recall":   round(recall_score(y_test, y_pred, average=avg, zero_division=0)*100, 2),
                "roc_auc":         roc_auc,
                "fit_status":      fit_status,
                "fit_warning":     fit_msg,
                "confusion_matrix": confusion_matrix(y_test, y_pred, labels=test_labels_idx).tolist(),
                "labels":          cm_labels,
                "class_report":    report,
                "loss_curve":      loss_curve,
                "val_curve":       val_curve,
                "preprocess_stats": {
                    "total_rows": len(df), "train_rows": len(X_train),
                    "test_rows": len(X_test), "n_features": X.shape[1],
                    "missing_filled": total_miss, "scaler": "StandardScaler",
                    "n_classes": n_cls, "class_names": le.classes_.tolist(),
                    "architecture": req.hidden_layers,
                },
                "n_iter":  mdl.n_iter_,
                "n_train": len(X_train),
                "n_test":  len(X_test),
            })

        else:
            y = pd.to_numeric(df[req.target], errors='coerce').fillna(0)

            yield emit("preprocess", 42, f"Split {int((1-req.test_size)*100)}/{int(req.test_size*100)}%...")
            await asyncio.sleep(0.1)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=req.test_size, random_state=42)

            yield emit("train", 50, f"Training MLP Regressor [{arch}] · {req.activation}...")
            await asyncio.sleep(0.3)

            mdl = MLPRegressor(
                hidden_layer_sizes=hidden, activation=req.activation,
                max_iter=req.max_iter, random_state=42,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=15, learning_rate='adaptive', alpha=0.001,
            )
            mdl.fit(X_train, y_train)

            yield emit("train", 70, f"Converged in {mdl.n_iter_} iters. Computing metrics...")
            await asyncio.sleep(0.1)

            y_pred       = mdl.predict(X_test)
            y_pred_train = mdl.predict(X_train)
            mse      = mean_squared_error(y_test, y_pred)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2  = r2_score(y_test, y_pred)
            fit_status, fit_msg = overfitting_status(max(0, train_r2), max(0, test_r2))

            mape      = float(np.mean(np.abs((y_test-y_pred)/(np.abs(y_test)+1e-8)))*100)
            med_ae    = float(median_absolute_error(y_test, y_pred))
            loss_curve = [round(float(v), 6) for v in mdl.loss_curve_]
            val_curve  = [round(float(v), 6) for v in (mdl.validation_scores_ or [])]
            scatter    = [{"actual": round(float(a),4), "predicted": round(float(p),4)}
                          for a,p in zip(y_test[:200], y_pred[:200])]

            db_save({
                "filename": req.filename, "n_rows": len(df), "n_cols": len(df.columns),
                "model_type": "neural-regression", "model_name": f"MLP {arch}",
                "metric_key": "r2", "metric_val": round(test_r2, 4),
                "summary": f"MLP Regressor [{arch}] · R² {round(test_r2,4)} · {mdl.n_iter_} iters",
            })

            yield emit("done", 100, "Training complete!", {
                "problem_type":   "regression",
                "mae":   round(float(mean_absolute_error(y_test,y_pred)), 4),
                "mse":   round(float(mse), 4),
                "rmse":  round(float(math.sqrt(mse)), 4),
                "r2":    round(float(test_r2), 4),
                "train_r2":  round(float(train_r2), 4),
                "mape":  round(mape, 2),
                "median_ae": round(med_ae, 4),
                "fit_status":  fit_status,
                "fit_warning": fit_msg,
                "loss_curve":  loss_curve,
                "val_curve":   val_curve,
                "scatter":     scatter,
                "preprocess_stats": {
                    "total_rows": len(df), "train_rows": len(X_train),
                    "test_rows": len(X_test), "n_features": X.shape[1],
                    "missing_filled": total_miss, "scaler": "StandardScaler",
                    "architecture": req.hidden_layers,
                },
                "n_iter":  mdl.n_iter_,
                "n_train": len(X_train),
                "n_test":  len(X_test),
            })

    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ─────────────────────────────────────────────
# CODE EXPORT
# ─────────────────────────────────────────────
class CodeRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    model_name: str
    target: Optional[str] = None
    features: List[str]
    test_size: Optional[float] = 0.2
    n_clusters: Optional[int] = 3
    eps: Optional[float] = 0.5
    min_samples: Optional[int] = 5
    problem_type: Optional[str] = 'classification'
    hidden_layers: Optional[List[int]] = None
    activation: Optional[str] = 'relu'
    max_iter: Optional[int] = 200

@app.post('/generate-code')
async def generate_code(req: CodeRequest):
    fs = str(req.features)

    if req.model_type == 'classification':
        model_map = {
            'logistic_regression': ("LogisticRegression(max_iter=2000,C=1.0,class_weight='balanced')", 'from sklearn.linear_model import LogisticRegression'),
            'decision_tree':       ("DecisionTreeClassifier(random_state=42,max_depth=6,min_samples_leaf=3,class_weight='balanced')", 'from sklearn.tree import DecisionTreeClassifier'),
            'random_forest':       ("RandomForestClassifier(n_estimators=200,random_state=42,max_depth=10,class_weight='balanced')", 'from sklearn.ensemble import RandomForestClassifier'),
            'svm':                 ("SVC(probability=True,random_state=42,kernel='rbf',class_weight='balanced')", 'from sklearn.svm import SVC'),
            'knn':                 ("KNeighborsClassifier(n_neighbors=7,weights='distance')", 'from sklearn.neighbors import KNeighborsClassifier'),
        }
        me, mi = model_map.get(req.model_name, model_map['random_forest'])
        code = f"""# ML Platform — Classification · {req.model_name.replace('_',' ').title()}
# Target: {req.target}
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
{mi}

df = pd.read_csv('your_dataset.csv')
features, target = {fs}, '{req.target}'
X = df[features].select_dtypes(include=[np.number])
y = LabelEncoder().fit_transform(df[target].astype(str))

# Safe stratify
strat = y if min(Counter(y).values()) >= 2 else None
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size={req.test_size}, random_state=42, stratify=strat)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler()),
    ('model',   {me}),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

avg = 'weighted' if len(set(y))>2 else 'binary'
cv  = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print(f"CV:       {{cv.mean()*100:.1f}}% ± {{cv.std()*100:.1f}}%")
print(f"Accuracy: {{accuracy_score(y_test,y_pred)*100:.2f}}%")
print(f"F1:       {{f1_score(y_test,y_pred,average=avg,zero_division=0)*100:.2f}}%")
print(classification_report(y_test, y_pred))

fig,axes=plt.subplots(1,3,figsize=(18,5))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d',cmap='Blues',ax=axes[0])
axes[0].set(title='Confusion Matrix',xlabel='Predicted',ylabel='Actual')
mo=pipe.named_steps['model']
if hasattr(mo,'feature_importances_'):
    pd.Series(mo.feature_importances_,index=X.columns).sort_values(ascending=True).plot(kind='barh',ax=axes[1])
    axes[1].set_title('Feature Importance')
axes[2].bar(range(1,len(cv)+1),cv*100,color='steelblue',alpha=0.8)
axes[2].axhline(cv.mean()*100,color='red',linestyle='--',label=f'Mean {{cv.mean()*100:.1f}}%')
axes[2].set(title='CV per Fold',ylabel='Accuracy (%)',ylim=(0,115)); axes[2].legend()
plt.tight_layout(); plt.savefig('classification_results.png',dpi=150); plt.show()
"""

    elif req.model_type == 'regression':
        model_map = {
            'linear_regression': ('LinearRegression()', 'from sklearn.linear_model import LinearRegression'),
            'ridge':             ('Ridge(alpha=1.0)', 'from sklearn.linear_model import Ridge'),
            'lasso':             ('Lasso(alpha=0.1,max_iter=5000)', 'from sklearn.linear_model import Lasso'),
            'decision_tree':     ('DecisionTreeRegressor(random_state=42,max_depth=6)', 'from sklearn.tree import DecisionTreeRegressor'),
            'random_forest':     ('RandomForestRegressor(n_estimators=200,random_state=42)', 'from sklearn.ensemble import RandomForestRegressor'),
        }
        me, mi = model_map.get(req.model_name, model_map['random_forest'])
        code = f"""# ML Platform — Regression · {req.model_name.replace('_',' ').title()}
# Target: {req.target}
import pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
{mi}

df = pd.read_csv('your_dataset.csv')
features, target = {fs}, '{req.target}'
X = df[features].select_dtypes(include=[np.number])
y = pd.to_numeric(df[target],errors='coerce').dropna(); X=X.loc[y.index]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size={req.test_size},random_state=42)
pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',RobustScaler()),('model',{me})])
pipe.fit(X_train,y_train); y_pred=pipe.predict(X_test)

cv=cross_val_score(pipe,X,y,cv=5,scoring='r2')
mse=mean_squared_error(y_test,y_pred)
print(f"CV R²:  {{cv.mean():.4f}} ± {{cv.std():.4f}}")
print(f"R²:     {{r2_score(y_test,y_pred):.4f}}")
print(f"MAE:    {{mean_absolute_error(y_test,y_pred):.4f}}")
print(f"RMSE:   {{math.sqrt(mse):.4f}}")

fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].scatter(y_test,y_pred,alpha=0.6,s=15)
mn,mx=min(float(y_test.min()),float(y_pred.min())),max(float(y_test.max()),float(y_pred.max()))
axes[0].plot([mn,mx],[mn,mx],'r--',lw=2); axes[0].set(title=f'Pred vs Actual (R²={{r2_score(y_test,y_pred):.3f}})',xlabel='Actual',ylabel='Predicted')
res=np.array(y_test)-y_pred; axes[1].scatter(y_pred,res,alpha=0.6,color='orange',s=15); axes[1].axhline(0,color='red',linestyle='--'); axes[1].set(title='Residuals',xlabel='Predicted',ylabel='Residual')
mo=pipe.named_steps['model']
if hasattr(mo,'feature_importances_'): pd.Series(mo.feature_importances_,index=X.columns).sort_values(ascending=True).plot(kind='barh',ax=axes[2]); axes[2].set_title('Feature Importance')
plt.tight_layout(); plt.savefig('regression_results.png',dpi=150); plt.show()
"""

    elif req.model_type == 'clustering':
        model_map = {
            'kmeans':        (f'KMeans(n_clusters={req.n_clusters},random_state=42,n_init="auto")', 'from sklearn.cluster import KMeans'),
            'dbscan':        (f'DBSCAN(eps={req.eps},min_samples={req.min_samples})', 'from sklearn.cluster import DBSCAN'),
            'agglomerative': (f'AgglomerativeClustering(n_clusters={req.n_clusters})', 'from sklearn.cluster import AgglomerativeClustering'),
        }
        me, mi = model_map.get(req.model_name, model_map['kmeans'])
        code = f"""# ML Platform — Clustering · {req.model_name.replace('_',' ').title()}
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
{mi}

df=pd.read_csv('your_dataset.csv'); features={fs}
X_raw=df[features].select_dtypes(include=[np.number])
X_imp=SimpleImputer(strategy='median').fit_transform(X_raw)
X_scaled=RobustScaler().fit_transform(X_imp)

model={me}; labels=model.fit_predict(X_scaled)
valid=[l for l in set(labels) if l!=-1]
print(f"Clusters: {{len(valid)}}  Noise: {{list(labels).count(-1)}}")
if len(valid)>=2:
    mask=np.array(labels)!=-1; Xv,lv=X_scaled[mask],np.array(labels)[mask]
    print(f"Silhouette:        {{silhouette_score(Xv,lv):.4f}}")
    print(f"Davies-Bouldin:    {{davies_bouldin_score(Xv,lv):.4f}}")
    print(f"Calinski-Harabasz: {{calinski_harabasz_score(Xv,lv):.1f}}")

COLORS=['#10b981','#6c63ff','#f59e0b','#0ea5e9','#f87171','#a78bfa']
pca=PCA(n_components=2); coords=pca.fit_transform(X_scaled); var=pca.explained_variance_ratio_
fig,axes=plt.subplots(1,2,figsize=(14,5))
for c in set(labels):
    m=np.array(labels)==c
    axes[0].scatter(coords[m,0],coords[m,1],c='#aaa' if c==-1 else COLORS[c%len(COLORS)],
        label='Noise' if c==-1 else f'C{{c}}',alpha=0.7,s=15)
axes[0].set(title=f'PCA ({{var[0]*100:.1f}}%+{{var[1]*100:.1f}}%)',xlabel='PC1',ylabel='PC2'); axes[0].legend(fontsize=8)
cnt=Counter(labels); lbl=['Noise' if k==-1 else f'C{{k}}' for k in sorted(cnt)]
axes[1].bar(lbl,[cnt[k] for k in sorted(cnt)],color=['#aaa' if k==-1 else COLORS[k%len(COLORS)] for k in sorted(cnt)])
axes[1].set(title='Cluster Sizes',ylabel='Count')
plt.tight_layout(); plt.savefig('clustering_results.png',dpi=150); plt.show()
"""

    elif req.model_type == 'neural':
        hidden = str(tuple(req.hidden_layers)) if req.hidden_layers else '(64, 32)'
        if req.problem_type == 'classification':
            code = f"""# ML Platform — Neural Network Classification
# Architecture: {req.hidden_layers} · Activation: {req.activation} · Target: {req.target}
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report

df=pd.read_csv('your_dataset.csv'); features,target={fs},'{req.target}'
X=df[features].select_dtypes(include=[np.number])
y=LabelEncoder().fit_transform(df[target].astype(str))
X=SimpleImputer(strategy='median').fit_transform(X); X=StandardScaler().fit_transform(X)
strat=y if min(Counter(y).values())>=2 else None
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size={req.test_size},random_state=42,stratify=strat)

model=MLPClassifier(hidden_layer_sizes={hidden},activation='{req.activation}',max_iter={req.max_iter},
    random_state=42,early_stopping=True,validation_fraction=0.15,
    n_iter_no_change=15,learning_rate='adaptive',alpha=0.001)
model.fit(X_train,y_train); y_pred=model.predict(X_test)
avg='weighted' if len(set(y))>2 else 'binary'
print(f"Iters:    {{model.n_iter_}}")
print(f"Accuracy: {{accuracy_score(y_test,y_pred)*100:.2f}}%")
print(f"F1:       {{f1_score(y_test,y_pred,average=avg,zero_division=0)*100:.2f}}%")
print(classification_report(y_test,y_pred))

fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].plot(model.loss_curve_,color='#f59e0b',lw=2,label='Loss')
if model.validation_scores_: axes[0].twinx().plot(model.validation_scores_,'#10b981',lw=2,linestyle='--')
axes[0].set(title='Loss Curve',xlabel='Epoch',ylabel='Loss')
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d',cmap='YlOrRd',ax=axes[1])
axes[1].set(title='Confusion Matrix',xlabel='Predicted',ylabel='Actual')
from sklearn.metrics import precision_score,recall_score
vals=[accuracy_score(y_test,y_pred)*100,f1_score(y_test,y_pred,average=avg,zero_division=0)*100,
      precision_score(y_test,y_pred,average=avg,zero_division=0)*100,recall_score(y_test,y_pred,average=avg,zero_division=0)*100]
bars=axes[2].bar(['Acc','F1','Prec','Rec'],vals,color=['#6c63ff','#0ea5e9','#10b981','#f59e0b'],alpha=0.85)
for b,v in zip(bars,vals): axes[2].text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{{v:.1f}}%',ha='center',fontsize=10)
axes[2].set(title='Metrics',ylim=(0,115))
plt.tight_layout(); plt.savefig('neural_clf.png',dpi=150); plt.show()
"""
        else:
            code = f"""# ML Platform — Neural Network Regression
# Architecture: {req.hidden_layers} · Activation: {req.activation} · Target: {req.target}
import pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df=pd.read_csv('your_dataset.csv'); features,target={fs},'{req.target}'
X=df[features].select_dtypes(include=[np.number])
y=pd.to_numeric(df[target],errors='coerce').fillna(0)
X=SimpleImputer(strategy='median').fit_transform(X); X=StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size={req.test_size},random_state=42)

model=MLPRegressor(hidden_layer_sizes={hidden},activation='{req.activation}',max_iter={req.max_iter},
    random_state=42,early_stopping=True,validation_fraction=0.15,
    n_iter_no_change=15,learning_rate='adaptive',alpha=0.001)
model.fit(X_train,y_train); y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
print(f"R²: {{r2_score(y_test,y_pred):.4f}}  MAE: {{mean_absolute_error(y_test,y_pred):.4f}}  RMSE: {{math.sqrt(mse):.4f}}")

fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].plot(model.loss_curve_,color='#f59e0b',lw=2); axes[0].set(title='Loss',xlabel='Epoch',ylabel='Loss')
axes[1].scatter(y_test,y_pred,alpha=0.6,s=15)
mn,mx=min(float(y_test.min()),float(y_pred.min())),max(float(y_test.max()),float(y_pred.max()))
axes[1].plot([mn,mx],[mn,mx],'r--',lw=2); axes[1].set(title=f'Pred vs Actual (R²={{r2_score(y_test,y_pred):.3f}})',xlabel='Actual',ylabel='Predicted')
res=np.array(y_test)-y_pred; axes[2].scatter(y_pred,res,alpha=0.6,color='orange',s=15); axes[2].axhline(0,color='red',linestyle='--')
axes[2].set(title='Residuals',xlabel='Predicted',ylabel='Residual')
plt.tight_layout(); plt.savefig('neural_reg.png',dpi=150); plt.show()
"""
    else:
        code = "# Unknown model type"

    return {'code': code}