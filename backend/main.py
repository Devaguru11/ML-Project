from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import io
import math

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
    r2_score, silhouette_score
)

from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared preprocessing pipeline ──────────────────────────
def make_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('model', model),
    ])

def overfitting_check(train_score, test_score, threshold=0.15):
    gap = train_score - test_score
    if gap > threshold:
        return f"Warning: model may be overfitting (train {train_score:.1%} vs test {test_score:.1%}). Try a simpler model or more data."
    if test_score < 0.5 and train_score < 0.55:
        return "Warning: model is underfitting. Try a more complex model or add more features."
    return None

# ===========================
# UPLOAD
# ===========================
@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if len(df) < 10:
        raise HTTPException(400, "Dataset must have at least 10 rows.")
    columns = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]
    return {"columns": columns, "rows": len(df)}

# ===========================
# ANALYSE
# ===========================
@app.post('/analyse')
async def analyse(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    num_df = df.select_dtypes(include=[np.number])

    histograms = {}
    for col in num_df.columns:
        counts, bin_edges = np.histogram(num_df[col].dropna(), bins=20)
        histograms[col] = {
            'counts': counts.tolist(),
            'bins': [round(float(b), 4) for b in bin_edges[:-1]]
        }

    corr = num_df.corr().round(2)
    correlation = {
        'columns': list(corr.columns),
        'matrix': corr.fillna(0).values.tolist()
    }

    scatter_data = num_df.head(200).fillna(0).to_dict(orient='records')

    missing = [
        {'column': col, 'missing': int(df[col].isnull().sum()),
         'pct': round(df[col].isnull().sum() / len(df) * 100, 1)}
        for col in df.columns
    ]

    boxplots = {}
    for col in num_df.columns:
        s = num_df[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        boxplots[col] = {
            'min': round(float(s.min()), 4),
            'q1': round(q1, 4),
            'median': round(float(s.median()), 4),
            'q3': round(q3, 4),
            'max': round(float(s.max()), 4),
            'outliers': [round(float(v), 4) for v in s[(s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)].head(50)]
        }

    return {
        'numeric_columns': list(num_df.columns),
        'histograms': histograms,
        'correlation': correlation,
        'scatter_data': scatter_data,
        'missing': missing,
        'boxplots': boxplots
    }

# ===========================
# CLASSIFICATION
# ===========================
class ClassifyRequest(BaseModel):
    model_name: str
    target: str
    features: List[str]
    test_size: float
    csv_data: str

@app.post('/classify')
async def classify(req: ClassifyRequest):
    import base64
    csv_bytes = base64.b64decode(req.csv_data)
    df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))

    # Use only selected feature columns
    available = [f for f in req.features if f in df.columns]
    X = df[available].select_dtypes(include=[np.number])
    y = df[req.target]

    if X.shape[1] == 0:
        raise HTTPException(400, "No numeric feature columns found.")
    if len(X) < 20:
        raise HTTPException(400, "Need at least 20 rows to train.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=req.test_size, random_state=42, stratify=y_enc
    )

    # ── Better model configs with tuned hyperparameters ──
    base_models = {
        'logistic_regression': LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', solver='lbfgs'),
        'decision_tree':       DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=3, class_weight='balanced'),
        'random_forest':       RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, min_samples_leaf=2, class_weight='balanced', n_jobs=-1),
        'svm':                 SVC(probability=True, random_state=42, kernel='rbf', C=1.0, class_weight='balanced'),
        'knn':                 KNeighborsClassifier(n_neighbors=min(7, len(X_train)//5 or 1), weights='distance', metric='minkowski'),
    }

    base = base_models.get(req.model_name)
    if not base:
        raise HTTPException(400, "Invalid model name.")

    pipe = make_pipeline(base)
    pipe.fit(X_train, y_train)

    y_pred       = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    avg = 'weighted' if n_classes > 2 else 'binary'
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred)

    # Cross-validation score for robustness
    try:
        cv_scores = cross_val_score(pipe, X, y_enc, cv=min(5, len(X)//10 or 2), scoring='accuracy')
        cv_mean   = round(float(cv_scores.mean()) * 100, 2)
        cv_std    = round(float(cv_scores.std())  * 100, 2)
    except Exception:
        cv_mean, cv_std = None, None

    fit_warning = overfitting_check(train_acc, test_acc)

    # Feature importance
    model_obj = pipe.named_steps['model']
    feature_importance = []
    if hasattr(model_obj, 'feature_importances_'):
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(X.columns, model_obj.feature_importances_)],
            key=lambda x: x["importance"], reverse=True
        )
    elif hasattr(model_obj, 'coef_'):
        coefs = np.abs(model_obj.coef_[0]) if model_obj.coef_.ndim > 1 else np.abs(model_obj.coef_)
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(X.columns, coefs)],
            key=lambda x: x["importance"], reverse=True
        )

    return {
        'accuracy':  round(test_acc * 100, 2),
        'f1':        round(f1_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        'recall':    round(recall_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        'train_accuracy': round(train_acc * 100, 2),
        'cv_accuracy': cv_mean,
        'cv_std': cv_std,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'labels': le.classes_.tolist(),
        'feature_importance': feature_importance,
        'fit_warning': fit_warning,
        'n_train': len(X_train),
        'n_test':  len(X_test),
    }

# ===========================
# REGRESSION
# ===========================
class RegressRequest(BaseModel):
    model_name: str
    target: str
    features: List[str]
    test_size: float
    csv_data: str

@app.post('/regress')
async def regress(req: RegressRequest):
    import base64
    csv_bytes = base64.b64decode(req.csv_data)
    df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))

    available = [f for f in req.features if f in df.columns]
    X = df[available].select_dtypes(include=[np.number])
    y = pd.to_numeric(df[req.target], errors='coerce')

    if X.shape[1] == 0:
        raise HTTPException(400, "No numeric feature columns found.")
    if y.isnull().all():
        raise HTTPException(400, "Target column has no numeric values.")

    valid = ~y.isnull()
    X, y = X[valid], y[valid]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=42)

    base_models = {
        'linear_regression': LinearRegression(),
        'ridge':             Ridge(alpha=1.0),
        'lasso':             Lasso(alpha=0.1, max_iter=5000),
        'decision_tree':     DecisionTreeRegressor(random_state=42, max_depth=6, min_samples_leaf=3),
        'random_forest':     RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10, min_samples_leaf=2, n_jobs=-1),
    }

    base = base_models.get(req.model_name)
    if not base:
        raise HTTPException(400, "Invalid model name.")

    pipe = make_pipeline(base)
    pipe.fit(X_train, y_train)

    y_pred       = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2  = r2_score(y_test,  y_pred)
    mse      = mean_squared_error(y_test, y_pred)

    fit_warning = overfitting_check(max(0, train_r2), max(0, test_r2))

    try:
        cv_scores = cross_val_score(pipe, X, y, cv=min(5, len(X)//10 or 2), scoring='r2')
        cv_mean   = round(float(cv_scores.mean()), 4)
        cv_std    = round(float(cv_scores.std()), 4)
    except Exception:
        cv_mean, cv_std = None, None

    model_obj = pipe.named_steps['model']
    feature_importance = []
    if hasattr(model_obj, 'feature_importances_'):
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(X.columns, model_obj.feature_importances_)],
            key=lambda x: x["importance"], reverse=True
        )
    elif hasattr(model_obj, 'coef_'):
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(abs(i)), 4)} for f, i in zip(X.columns, model_obj.coef_)],
            key=lambda x: x["importance"], reverse=True
        )

    scatter = [{"actual": round(float(a), 4), "predicted": round(float(p), 4)}
               for a, p in zip(y_test[:200], y_pred[:200])]

    return {
        'mae':        round(float(mean_absolute_error(y_test, y_pred)), 4),
        'mse':        round(float(mse), 4),
        'rmse':       round(float(math.sqrt(mse)), 4),
        'r2':         round(float(test_r2), 4),
        'train_r2':   round(float(train_r2), 4),
        'cv_r2':      cv_mean,
        'cv_std':     cv_std,
        'feature_importance': feature_importance,
        'scatter':    scatter,
        'fit_warning': fit_warning,
        'n_train': len(X_train),
        'n_test':  len(X_test),
    }

# ===========================
# CLUSTERING
# ===========================
class ClusterRequest(BaseModel):
    model_name: str
    features: List[str]
    csv_data: str
    n_clusters: Optional[int] = 3
    eps: Optional[float] = 0.5
    min_samples: Optional[int] = 5

@app.post('/cluster')
async def cluster(req: ClusterRequest):
    import base64
    from collections import Counter
    csv_bytes = base64.b64decode(req.csv_data)
    df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))

    available = [f for f in req.features if f in df.columns]
    X = df[available].select_dtypes(include=[np.number])

    imputer = SimpleImputer(strategy='median')
    X_imp   = imputer.fit_transform(X)
    scaler  = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)

    if req.model_name == 'kmeans':
        model = KMeans(n_clusters=req.n_clusters, random_state=42, n_init='auto', max_iter=500)
    elif req.model_name == 'dbscan':
        model = DBSCAN(eps=req.eps, min_samples=req.min_samples)
    elif req.model_name == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=req.n_clusters)
    else:
        raise HTTPException(400, "Invalid model")

    labels = model.fit_predict(X_scaled).tolist()
    unique_labels = set(labels)
    valid_labels  = [l for l in unique_labels if l != -1]

    silhouette = None
    if len(valid_labels) >= 2:
        mask = np.array(labels) != -1
        if mask.sum() > len(valid_labels):
            try:
                silhouette = round(float(silhouette_score(X_scaled[mask], np.array(labels)[mask])), 4)
            except Exception:
                silhouette = None

    pca    = PCA(n_components=min(2, X_scaled.shape[1]))
    coords = pca.fit_transform(X_scaled)

    scatter = [
        {"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4), "cluster": labels[i]}
        for i in range(len(labels))
    ]

    counts = Counter(labels)
    cluster_sizes = [
        {"cluster": int(k), "size": v, "label": "Noise" if k == -1 else f"Cluster {k}"}
        for k, v in sorted(counts.items())
    ]

    return {
        "labels": labels,
        "silhouette": silhouette,
        "scatter": scatter,
        "cluster_sizes": cluster_sizes,
        "n_clusters_found": len(valid_labels),
        "n_noise": labels.count(-1),
    }

# ===========================
# NEURAL NETWORK
# ===========================
class NeuralRequest(BaseModel):
    problem_type: str
    target: str
    features: List[str]
    test_size: float
    csv_data: str
    hidden_layers: List[int]
    activation: str
    max_iter: int

@app.post('/neural')
async def neural(req: NeuralRequest):
    import base64
    csv_bytes = base64.b64decode(req.csv_data)
    df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))

    available = [f for f in req.features if f in df.columns]
    X = df[available].select_dtypes(include=[np.number])
    hidden = tuple(req.hidden_layers) if req.hidden_layers else (100,)

    imputer  = SimpleImputer(strategy='median')
    X_imp    = imputer.fit_transform(X)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    if req.problem_type == 'classification':
        y   = df[req.target]
        le  = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=req.test_size, random_state=42, stratify=y_enc
        )
        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=req.activation,
            max_iter=req.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            learning_rate='adaptive',
            alpha=0.001,
        )
        model.fit(X_train, y_train)
        y_pred       = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        avg = 'weighted' if len(set(y_enc)) > 2 else 'binary'

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test, y_pred)
        fit_warning = overfitting_check(train_acc, test_acc)
        loss_curve = [round(float(v), 6) for v in model.loss_curve_]

        return {
            'problem_type': 'classification',
            'accuracy':  round(test_acc * 100, 2),
            'f1':        round(f1_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
            'precision': round(precision_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
            'recall':    round(recall_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
            'train_accuracy': round(train_acc * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'labels':    le.classes_.tolist(),
            'loss_curve': loss_curve,
            'fit_warning': fit_warning,
            'n_iter':  model.n_iter_,
            'n_train': len(X_train),
            'n_test':  len(X_test),
        }

    else:
        y = pd.to_numeric(df[req.target], errors='coerce').fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=req.test_size, random_state=42
        )
        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=req.activation,
            max_iter=req.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            learning_rate='adaptive',
            alpha=0.001,
        )
        model.fit(X_train, y_train)
        y_pred       = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        mse = mean_squared_error(y_test, y_pred)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2  = r2_score(y_test, y_pred)
        fit_warning = overfitting_check(max(0, train_r2), max(0, test_r2))
        loss_curve = [round(float(v), 6) for v in model.loss_curve_]
        scatter = [{"actual": round(float(a), 4), "predicted": round(float(p), 4)}
                   for a, p in zip(y_test[:200], y_pred[:200])]

        return {
            'problem_type': 'regression',
            'mae':  round(float(mean_absolute_error(y_test, y_pred)), 4),
            'mse':  round(float(mse), 4),
            'rmse': round(float(math.sqrt(mse)), 4),
            'r2':   round(float(test_r2), 4),
            'train_r2': round(float(train_r2), 4),
            'loss_curve': loss_curve,
            'scatter': scatter,
            'fit_warning': fit_warning,
            'n_iter':  model.n_iter_,
            'n_train': len(X_train),
            'n_test':  len(X_test),
        }

# ===========================
# CODE EXPORT
# ===========================
class CodeRequest(BaseModel):
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
    code = build_commented_code(req)
    return {'code': code}


def build_commented_code(req: CodeRequest) -> str:
    features_str = str(req.features)

    PREPROCESS = """
# ── Preprocessing pipeline (handles missing values + scaling) ──
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # fill missing values with column median
    ('scaler', RobustScaler()),                     # scale features, robust to outliers
])
"""

    if req.model_type == 'classification':
        model_map = {
            'logistic_regression': ("LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')", 'from sklearn.linear_model import LogisticRegression'),
            'decision_tree':       ("DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=3, class_weight='balanced')", 'from sklearn.tree import DecisionTreeClassifier'),
            'random_forest':       ("RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, class_weight='balanced')", 'from sklearn.ensemble import RandomForestClassifier'),
            'svm':                 ("SVC(probability=True, random_state=42, kernel='rbf', class_weight='balanced')", 'from sklearn.svm import SVC'),
            'knn':                 ("KNeighborsClassifier(n_neighbors=7, weights='distance')", 'from sklearn.neighbors import KNeighborsClassifier'),
        }
        model_expr, model_import = model_map.get(req.model_name, model_map['random_forest'])
        return f"""# ML Platform — Classification · {req.model_name.replace('_',' ').title()}
# Target: {req.target}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
{model_import}

df = pd.read_csv('your_dataset.csv')

features = {features_str}
target   = '{req.target}'

X = df[features].select_dtypes(include=[np.number])
y = df[target]

# Encode labels
le = LabelEncoder()
y  = le.fit_transform(y.astype(str))

# Stratified split — keeps class proportions equal in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={req.test_size}, random_state=42, stratify=y
)

# Pipeline: impute missing → scale → train
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler()),
    ('model',   {model_expr}),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Cross-validation (more reliable than single split)
cv = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print(f'CV Accuracy: {{cv.mean()*100:.1f}}% ± {{cv.std()*100:.1f}}%')

avg = 'weighted' if len(set(y)) > 2 else 'binary'
print(f'Test Accuracy:  {{accuracy_score(y_test, y_pred)*100:.2f}}%')
print(f'F1 Score:       {{f1_score(y_test, y_pred, average=avg, zero_division=0)*100:.2f}}%')
print(f'Precision:      {{precision_score(y_test, y_pred, average=avg, zero_division=0)*100:.2f}}%')
print(f'Recall:         {{recall_score(y_test, y_pred, average=avg, zero_division=0)*100:.2f}}%')
print('Labels:', le.classes_)
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
"""

    elif req.model_type == 'regression':
        model_map = {
            'linear_regression': ('LinearRegression()', 'from sklearn.linear_model import LinearRegression'),
            'ridge':             ('Ridge(alpha=1.0)', 'from sklearn.linear_model import Ridge'),
            'lasso':             ('Lasso(alpha=0.1, max_iter=5000)', 'from sklearn.linear_model import Lasso'),
            'decision_tree':     ('DecisionTreeRegressor(random_state=42, max_depth=6, min_samples_leaf=3)', 'from sklearn.tree import DecisionTreeRegressor'),
            'random_forest':     ('RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)', 'from sklearn.ensemble import RandomForestRegressor'),
        }
        model_expr, model_import = model_map.get(req.model_name, model_map['random_forest'])
        return f"""# ML Platform — Regression · {req.model_name.replace('_',' ').title()}
# Target: {req.target}

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
{model_import}

df = pd.read_csv('your_dataset.csv')

features = {features_str}
target   = '{req.target}'

X = df[features].select_dtypes(include=[np.number])
y = pd.to_numeric(df[target], errors='coerce').dropna()
X = X.loc[y.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={req.test_size}, random_state=42
)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler()),
    ('model',   {model_expr}),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

cv = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f'CV R2: {{cv.mean():.4f}} ± {{cv.std():.4f}}')

mse = mean_squared_error(y_test, y_pred)
print(f'MAE:  {{mean_absolute_error(y_test, y_pred):.4f}}')
print(f'RMSE: {{math.sqrt(mse):.4f}}')
print(f'R2:   {{r2_score(y_test, y_pred):.4f}}')
"""

    elif req.model_type == 'clustering':
        model_map = {
            'kmeans':        (f'KMeans(n_clusters={req.n_clusters}, random_state=42, n_init="auto")', 'from sklearn.cluster import KMeans'),
            'dbscan':        (f'DBSCAN(eps={req.eps}, min_samples={req.min_samples})', 'from sklearn.cluster import DBSCAN'),
            'agglomerative': (f'AgglomerativeClustering(n_clusters={req.n_clusters})', 'from sklearn.cluster import AgglomerativeClustering'),
        }
        model_expr, model_import = model_map.get(req.model_name, model_map['kmeans'])
        return f"""# ML Platform — Clustering · {req.model_name.replace('_',' ').title()}

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
{model_import}

df = pd.read_csv('your_dataset.csv')
features = {features_str}
X = df[features].select_dtypes(include=[np.number])

# Preprocess
imp = SimpleImputer(strategy='median')
X_imp = imp.fit_transform(X)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imp)

model  = {model_expr}
labels = model.fit_predict(X_scaled)

valid = [l for l in set(labels) if l != -1]
if len(valid) >= 2:
    mask = np.array(labels) != -1
    print(f'Silhouette Score: {{silhouette_score(X_scaled[mask], np.array(labels)[mask]):.4f}}')

print(f'Clusters found: {{len(valid)}}')
print(f'Noise points:   {{list(labels).count(-1)}}')

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
print('PCA shape for plotting:', coords.shape)
"""

    elif req.model_type == 'neural':
        hidden = str(tuple(req.hidden_layers)) if req.hidden_layers else '(64, 32)'
        clf_code = f"""# ML Platform — Neural Network (Classification)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('your_dataset.csv')
features = {features_str}
target   = '{req.target}'

X = df[features].select_dtypes(include=[np.number])
y = LabelEncoder().fit_transform(df[target].astype(str))

X = SimpleImputer(strategy='median').fit_transform(X)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={req.test_size}, random_state=42, stratify=y)

model = MLPClassifier(
    hidden_layer_sizes={hidden},
    activation='{req.activation}',
    max_iter={req.max_iter},
    random_state=42,
    early_stopping=True,
    learning_rate='adaptive',
    alpha=0.001,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

avg = 'weighted' if len(set(y)) > 2 else 'binary'
print(f'Accuracy:  {{accuracy_score(y_test, y_pred)*100:.2f}}%')
print(f'F1 Score:  {{f1_score(y_test, y_pred, average=avg, zero_division=0)*100:.2f}}%')
print(f'Iterations: {{model.n_iter_}}')
"""
        reg_code = f"""# ML Platform — Neural Network (Regression)

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('your_dataset.csv')
features = {features_str}
target   = '{req.target}'

X = df[features].select_dtypes(include=[np.number])
y = pd.to_numeric(df[target], errors='coerce').fillna(0)

X = SimpleImputer(strategy='median').fit_transform(X)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={req.test_size}, random_state=42)

model = MLPRegressor(
    hidden_layer_sizes={hidden},
    activation='{req.activation}',
    max_iter={req.max_iter},
    random_state=42,
    early_stopping=True,
    learning_rate='adaptive',
    alpha=0.001,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE:  {{mean_absolute_error(y_test, y_pred):.4f}}')
print(f'RMSE: {{math.sqrt(mse):.4f}}')
print(f'R2:   {{r2_score(y_test, y_pred):.4f}}')
"""
        return clf_code if req.problem_type == 'classification' else reg_code

    return "# No code generated"