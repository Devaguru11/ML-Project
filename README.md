# ML Platform

A comprehensive machine learning platform that simplifies the entire ML workflow from data upload to model deployment. Upload your dataset, get automatic analysis and model suggestions, preprocess data, train models with real-time progress, and export production-ready Python code.

## Features

- **📁 Multi-format Support**: Upload CSV, Excel, TSV, or JSON files
- **🔍 Auto-Analysis**: Automatic dataset health check and ML model suggestions
- **⚙️ Smart Preprocessing**: Handles missing values, outliers, duplicates, and scaling
- **🚀 Real-time Training**: Watch preprocessing, training, and evaluation stages live
- **📊 Comprehensive Metrics**: Accuracy, F1, ROC-AUC, confusion matrices, cross-validation, feature importance
- **💾 Code Export**: Generate clean, commented Python scripts for your trained models
- **🗄️ Training History**: SQLite-based history of all your experiments

## Supported ML Tasks

- **Classification**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN
- **Regression**: Linear, Ridge, Lasso, Decision Tree, Random Forest
- **Clustering**: K-Means, DBSCAN, Agglomerative Clustering
- **Neural Networks**: Multi-layer Perceptron (MLP) for classification and regression

## Tech Stack

### Backend
- **FastAPI**: High-performance async web framework
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Pandas & NumPy**: Data manipulation and analysis
- **SQLite**: Local database for training history
- **Uvicorn**: ASGI server

### Frontend
- **React 19**: Modern UI framework with hooks
- **Vite**: Fast build tool and dev server
- **React Router**: Client-side routing
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Declarative charting library

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

   The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

### Usage

1. Open your browser and go to `http://localhost:5173`
2. Upload your dataset (CSV, Excel, TSV, or JSON)
3. Review the automatic dataset analysis and model suggestions
4. Preprocess your data if needed
5. Choose your ML task and configure the model
6. Train the model and watch the real-time progress
7. Analyze the results and export Python code

## API Endpoints

The backend provides the following REST API endpoints:

- `POST /upload` - Upload and parse dataset files
- `POST /analyse` - Generate visualizations and statistics
- `POST /analyse-dataset` - Deep dataset analysis with model suggestions
- `POST /preprocess` - Clean and prepare dataset
- `POST /classify` - Train classification models (streaming)
- `POST /regress` - Train regression models (streaming)
- `POST /cluster` - Perform clustering (streaming)
- `POST /neural` - Train neural networks (streaming)
- `POST /generate-code` - Export Python code for trained models
- `GET /history` - Get training history
- `DELETE /history` - Clear training history

## Project Structure

```
ML-Project/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── ml_platform.db       # SQLite database (created automatically)
│   └── ...
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React app
│   │   ├── api/client.js    # API client functions
│   │   ├── pages/           # React pages for each step
│   │   └── components/      # Reusable UI components
│   ├── package.json         # Node dependencies
│   └── ...
└── README.md                # This file
```

## Development

### Backend Development
- The server auto-reloads on code changes when using `--reload`
- API documentation available at `http://localhost:8000/docs` (Swagger UI)

### Frontend Development
- Hot module replacement enabled for instant updates
- ESLint configured for code quality
- Tailwind CSS for styling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.