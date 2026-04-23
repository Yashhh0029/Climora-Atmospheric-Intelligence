# Climora: Atmospheric Intelligence
> A Context-Aware, Self-Adaptive, Behavior-Aware Forecasting Engine

Climora is a high-performance, production-grade weather prediction platform. Moving far beyond traditional static regression, Climora leverages a custom-built Hybrid Machine Learning pipeline (Ridge Baseline + RandomForest Residuals) fused with an advanced self-learning intelligence layer to deliver unparalleled, context-aware meteorological forecasts.

## 🚀 Core Intelligence Capabilities

1. **Context-Aware Analytics Layer**
   - Automatically detects structural dataset anomalies using Z-scores.
   - Computes historical temperature variance to classify data volatility.
   - Mathematically classifies historical patterns as Stable, Increasing, Decreasing, or Fluctuating.

2. **Behavior-Aware Prediction Layer**
   - **Regime Shift Detection:** Accurately identifies the exact point where a dataset's underlying trend reverses.
   - **Stability Guards:** Automatically clips final predictions within robust physical boundaries (`mean ± 3*std_dev`) to completely prevent unrealistic drift or exponential explosion from outliers.
   - **Monotonic Confidence Generation:** Calculates prediction confidence dynamically for each day, ensuring logic-bound monotonic decay across the forecasting horizon.

3. **Self-Adaptive Feedback System (Dataset Scoped)**
   - **Isolated Memory:** Hashes every uploaded CSV (SHA-256) to create completely isolated, persistent memory states for independent datasets.
   - **Autonomous Bias Correction:** Constantly tracks its own residuals (`actual - predicted`). If a systematic overprediction or underprediction bias is detected, the engine mathematically calibrates the final forecast outputs on the fly.
   - **Self-Improvement Tracking:** Calculates an ongoing Mean Absolute Error (MAE) improvement score to track if its forecasting accuracy is getting better or degrading over time.

4. **Lightning-Fast Performance**
   - Despite analyzing variance, detecting regime shifts, tracking disk-cached residuals, and running a Hybrid RF Model, the entire pipeline is heavily vectorized with NumPy.
   - Average execution time (from upload to completed payload delivery): **< 0.5 seconds**.

## 💻 Tech Stack
- **Backend:** Python, Flask, Waitress
- **Machine Learning:** Scikit-Learn (RidgeCV, RandomForestRegressor)
- **Math & Analytics:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** Vanilla JS, HTML5 Canvas (60FPS Particle Rendering Engine), CSS3

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/climora.git
   cd climora
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install flask pandas numpy scikit-learn matplotlib seaborn waitress joblib flask-cors
   ```

4. **Set Environment Variables:**
   - `FLASK_SECRET_KEY`: (Required) Set this to a secure random string.
   - `MODEL_VERSION`: (Optional) Defaults to `6.0`.
   - `PORT`: (Optional) Defaults to `5000`.

   *Windows PowerShell:*
   ```powershell
   $env:FLASK_SECRET_KEY="super_secret_production_key"
   ```
   *Linux/macOS:*
   ```bash
   export FLASK_SECRET_KEY="super_secret_production_key"
   ```

5. **Run the Engine:**

   *Windows (from project root):*
   ```cmd
   cmd /c "set FLASK_SECRET_KEY=your_secret_key && cd backend && ..\.venv\Scripts\python.exe app.py"
   ```
   *Linux/macOS (from project root):*
   ```bash
   export FLASK_SECRET_KEY="your_secret_key"
   cd backend
   python3 app.py
   ```
   *The application will boot using Waitress for production-ready WSGI serving on port 5000.*

## 📂 Project Structure

```
Climora/
├── backend/                  # Python server & ML engine
│   ├── app.py                # Entry point (app factory + server boot)
│   ├── config.py             # Environment variable configuration
│   ├── routes.py             # Flask route handlers
│   ├── middleware.py         # Rate limiting, request IDs, metrics
│   ├── requirements.txt      # Python dependencies
│   ├── ml/
│   │   ├── hybrid_model.py   # HybridWeatherModel (Ridge + RandomForest)
│   │   └── intelligence.py   # Analytics, regime detection, feedback loop
│   ├── utils/
│   │   ├── logger.py         # JSON structured logger
│   │   ├── cache.py          # Feedback history persistence
│   │   └── metrics.py        # System metrics & threading locks
│   └── visualization/
│       └── plot.py           # Matplotlib chart generation
├── frontend/                 # Static assets & HTML templates
│   ├── templates/            # Jinja2 HTML (index, result)
│   └── static/               # CSS & JS (60FPS canvas engine)
├── .gitignore
├── Procfile                  # Deployment config (Waitress)
└── README.md
```

> **Note:** `backend/model_cache/` and `backend/dataset/` are runtime-generated and excluded from git.
