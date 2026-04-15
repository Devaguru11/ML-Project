from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import io
import json

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

from pydantic import BaseModel
from typing import List

# ✅ CREATE APP
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# ANALYSE ENDPOINT
# ===========================
@app.post('/upload')
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    # Send column info to frontend
    columns = [
        {
            "name": col,
            "dtype": str(df[col].dtype)
        }
        for col in df.columns
    ]

    return {
        "columns": columns,
        "rows": len(df)
    }
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
            'outliers': [
                round(float(v), 4)
                for v in s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)].head(50)
            ]
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=42
    )

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

    return {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'f1': round(f1_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'labels': le.classes_.tolist(),
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
    import base64, math

    csv_bytes = base64.b64decode(req.csv_data)
    df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))

    X = df[req.features].select_dtypes(include=[np.number]).fillna(0)
    y = pd.to_numeric(df[req.target], errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=42
    )

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

    return {
        'mae': round(mean_absolute_error(y_test, y_pred), 4),
        'mse': round(mse, 4),
        'rmse': round(math.sqrt(mse), 4),
        'r2': round(r2_score(y_test, y_pred), 4),
    }