from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import io
import math

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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

# ===========================
# UPLOAD
# ===========================
@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
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
        {
            'column': col,
            'missing': int(df[col].isnull().sum()),
            'pct': round(df[col].isnull().sum() / len(df) * 100, 1)
        }
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
            'outliers': [round(float(v), 4) for v in s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)].head(50)]
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

    X = df[req.features].select_dtypes(include=[np.number]).fillna(0)
    y = df[req.target]
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=42)

    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'knn': KNeighborsClassifier(),
    }

    model = models.get(req.model_name)
    if not model:
        raise HTTPException(status_code=400, detail="Invalid model")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    avg = 'weighted' if len(set(y)) > 2 else 'binary'

    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(X.columns, model.feature_importances_)],
            key=lambda x: x["importance"], reverse=True
        )
    elif hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(X.columns, coefs)],
            key=lambda x: x["importance"], reverse=True
        )

    return {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'f1': round(f1_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'labels': le.classes_.tolist(),
        'feature_importance': feature_importance,
        'n_train': len(X_train),
        'n_test': len(X_test),
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

    X = df[req.features].select_dtypes(include=[np.number]).fillna(0)
    y = pd.to_numeric(df[req.target], errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size, random_state=42)

    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso(),
        'decision_tree': DecisionTreeRegressor(),
        'random_forest': RandomForestRegressor(),
    }

    model = models.get(req.model_name)
    if not model:
        raise HTTPException(status_code=400, detail="Invalid model")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(i), 4)} for f, i in zip(X.columns, model.feature_importances_)],
            key=lambda x: x["importance"], reverse=True
        )
    elif hasattr(model, 'coef_'):
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(abs(i)), 4)} for f, i in zip(X.columns, model.coef_)],
            key=lambda x: x["importance"], reverse=True
        )

    scatter = [{"actual": round(float(a), 4), "predicted": round(float(p), 4)}
               for a, p in zip(y_test[:200], y_pred[:200])]

    return {
        'mae': round(mean_absolute_error(y_test, y_pred), 4),
        'mse': round(mse, 4),
        'rmse': round(math.sqrt(mse), 4),
        'r2': round(r2_score(y_test, y_pred), 4),
        'feature_importance': feature_importance,
        'scatter': scatter,
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

# ===========================
# CLUSTERING
# ===========================
class ClusterRequest(BaseModel):
    model_name: str          # 'kmeans' | 'dbscan' | 'agglomerative'
    features: List[str]
    csv_data: str
    # K-Means / Agglomerative
    n_clusters: Optional[int] = 3
    # DBSCAN
    eps: Optional[float] = 0.5
    min_samples: Optional[int] = 5

@app.post('/cluster')
async def cluster(req: ClusterRequest):
    import base64
    csv_bytes = base64.b64decode(req.csv_data)
    df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))

    X = df[req.features].select_dtypes(include=[np.number]).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if req.model_name == 'kmeans':
        model = KMeans(n_clusters=req.n_clusters, random_state=42, n_init='auto')
    elif req.model_name == 'dbscan':
        model = DBSCAN(eps=req.eps, min_samples=req.min_samples)
    elif req.model_name == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=req.n_clusters)
    else:
        raise HTTPException(status_code=400, detail="Invalid model")

    labels = model.fit_predict(X_scaled).tolist()

    # Silhouette score (needs at least 2 clusters and no noise-only)
    unique_labels = set(labels)
    valid_labels = [l for l in unique_labels if l != -1]
    silhouette = None
    if len(valid_labels) >= 2:
        mask = np.array(labels) != -1
        if mask.sum() > len(valid_labels):
            try:
                silhouette = round(float(silhouette_score(X_scaled[mask], np.array(labels)[mask])), 4)
            except Exception:
                silhouette = None

    # PCA → 2D for scatter plot
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    scatter = [
        {"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4), "cluster": labels[i]}
        for i in range(len(labels))
    ]

    # Cluster size breakdown
    from collections import Counter
    counts = Counter(labels)
    cluster_sizes = [
        {"cluster": int(k), "size": v, "label": f"Noise" if k == -1 else f"Cluster {k}"}
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
    problem_type: str        # 'classification' | 'regression'
    target: str
    features: List[str]
    test_size: float
    csv_data: str
    hidden_layers: List[int] # e.g. [64, 32]
    activation: str          # 'relu' | 'tanh' | 'logistic'
    max_iter: int

@app.post('/neural')
async def neural(req: NeuralRequest):
    import base64
    csv_bytes = base64.b64decode(req.csv_data)
    df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))

    X = df[req.features].select_dtypes(include=[np.number]).fillna(0)
    hidden = tuple(req.hidden_layers) if req.hidden_layers else (100,)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if req.problem_type == 'classification':
        y = df[req.target]
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=req.test_size, random_state=42)

        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=req.activation,
            max_iter=req.max_iter,
            random_state=42,
            early_stopping=True,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        avg = 'weighted' if len(set(y_enc)) > 2 else 'binary'

        loss_curve = [round(float(v), 6) for v in model.loss_curve_]

        return {
            'problem_type': 'classification',
            'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
            'f1': round(f1_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
            'precision': round(precision_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
            'recall': round(recall_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'labels': le.classes_.tolist(),
            'loss_curve': loss_curve,
            'n_iter': model.n_iter_,
            'n_train': len(X_train),
            'n_test': len(X_test),
        }

    else:  # regression
        y = pd.to_numeric(df[req.target], errors='coerce').fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=req.test_size, random_state=42)

        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=req.activation,
            max_iter=req.max_iter,
            random_state=42,
            early_stopping=True,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        loss_curve = [round(float(v), 6) for v in model.loss_curve_]
        scatter = [{"actual": round(float(a), 4), "predicted": round(float(p), 4)}
                   for a, p in zip(y_test[:200], y_pred[:200])]

        return {
            'problem_type': 'regression',
            'mae': round(mean_absolute_error(y_test, y_pred), 4),
            'mse': round(mse, 4),
            'rmse': round(math.sqrt(mse), 4),
            'r2': round(r2_score(y_test, y_pred), 4),
            'loss_curve': loss_curve,
            'scatter': scatter,
            'n_iter': model.n_iter_,
            'n_train': len(X_train),
            'n_test': len(X_test),
        }