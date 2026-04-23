# 🌩️ CLIMORA
### An Atmospheric Intelligence Engine — Context-Aware, Self-Adaptive, Behavior-Aware Weather Forecasting

> **Climora is not a weather app.**  
> It is a self-learning, behavior-aware meteorological intelligence system engineered to predict, adapt, and evolve — not just display forecasts.

---

## 🌌 Vision

Climora is a **next-generation atmospheric prediction platform** built around **data-centric intelligence**, not traditional static models.

Unlike conventional weather APIs, Climora:
- Understands historical trends, volatility, and regime shifts
- Maintains self-learning feedback loops per dataset
- Operates entirely on uploaded CSV data — no external API dependency
- Acts as a true intelligent forecasting engine, not just a regression wrapper

This project is engineered with **production-grade architecture**, **hybrid ML pipelines**, and **real-time analytics**, targeting a **research-level design philosophy**.

---

## 🧠 Core Design Philosophy

| Principle | Description |
|-----------|-------------|
| Context-Aware Analytics | Automatic anomaly detection, variance classification, and trend recognition |
| Behavior-Aware Prediction | Regime shift detection with stability-guarded, physically bounded outputs |
| Self-Adaptive Feedback | Per-dataset isolated memory with autonomous bias correction on every run |
| Hybrid ML Architecture | Ridge regression baseline fused with RandomForest residual learning |
| Monotonic Confidence | Dynamically computed, logic-bound confidence decay across forecast horizon |
| Lightning Performance | Fully vectorized NumPy pipeline — end-to-end in under 0.5 seconds |

---

## 🏗️ System Architecture (High Level)

```text
CSV Upload
   ↓
Analytics Layer
(Anomaly Detection + Variance + Trend Classification)
   ↓
Hybrid ML Core
(Ridge Baseline + RandomForest Residuals)
   ↓
Intelligence Layer
(Regime Shift Detection + Stability Guards + Confidence Engine)
   ↓
Self-Adaptive Feedback Loop
(SHA-256 Dataset Memory + Bias Correction + MAE Tracking)
   ↓
Forecast Output + Visualization
(Interactive Charts + Structured JSON Payload)
```

---

## 🚀 Key Features

### 📡 Context-Aware Analytics Layer
- Automatically detects structural dataset anomalies using Z-scores
- Computes historical temperature variance to classify data volatility
- Mathematically classifies historical patterns as Stable, Increasing, Decreasing, or Fluctuating

### 🧠 Behavior-Aware Prediction Layer
- **Regime Shift Detection:** Identifies the exact point where a dataset's underlying trend reverses
- **Stability Guards:** Clips final predictions within robust physical boundaries (`mean ± 3×std_dev`) to prevent unrealistic drift
- **Monotonic Confidence Generation:** Dynamically computed confidence scores with logic-bound decay per forecast day

### 🔄 Self-Adaptive Feedback System (Dataset Scoped)
- **Isolated Memory:** SHA-256 hashes every uploaded CSV for completely independent persistent memory states
- **Autonomous Bias Correction:** Tracks residuals (`actual - predicted`) and auto-calibrates forecasts when systematic bias is detected
- **Self-Improvement Tracking:** Calculates ongoing MAE improvement score to monitor if forecasting accuracy is getting better or degrading

### ⚡ Lightning-Fast Performance
- Fully vectorized with NumPy — runs anomaly detection, regime shift analysis, disk-cached residual tracking, and Hybrid RF inference in **< 0.5 seconds**

### 🎨 60FPS Particle Visualization Frontend
- Real-time HTML5 Canvas particle rendering engine
- Context-aware weather animations that adapt to forecast conditions
- Fully responsive, glassmorphism-styled dark UI

---

## 🧩 Project Structure

```text
Climora/
├── backend/                    # Python server & ML intelligence engine
│   ├── app.py                  # Entry point (app factory + Waitress boot)
│   ├── config.py               # Environment variable configuration
│   ├── routes.py               # Flask route handlers
│   ├── middleware.py           # Rate limiting, request IDs, metrics
│   ├── requirements.txt        # Python dependencies
│   ├── ml/
│   │   ├── hybrid_model.py     # HybridWeatherModel (Ridge + RandomForest)
│   │   └── intelligence.py     # Analytics, regime detection, feedback loop
│   ├── utils/
│   │   ├── logger.py           # JSON structured logger
│   │   ├── cache.py            # Feedback history persistence (disk)
│   │   └── metrics.py          # System metrics & threading locks
│   └── visualization/
│       └── plot.py             # Matplotlib chart generation
├── frontend/                   # Static assets & Jinja2 HTML templates
│   ├── templates/
│   │   ├── index.html          # Upload & control interface
│   │   └── result.html         # Forecast results & visualization
│   └── static/
│       ├── style.css           # Global design system
│       ├── weather.css         # Weather-specific UI components
│       └── js/
│           └── weatherSystem.js  # 60FPS Canvas particle engine
├── .gitignore
├── Procfile                    # Deployment config (Waitress WSGI)
└── README.md
```

> **Note:** `backend/model_cache/` and `backend/dataset/` are runtime-generated and excluded from git.

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Backend Framework | Flask + Waitress (production WSGI) |
| Machine Learning | Scikit-Learn (RidgeCV, RandomForestRegressor) |
| Math & Analytics | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Frontend | Vanilla JS, HTML5 Canvas, CSS3 |
| Intelligence | Custom regime detection, bias correction & confidence engine |
| Platform | Cross-platform (Windows / Linux / macOS) |

---

## ▶️ How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/Yashhh0029/Climora-Atmospheric-Intelligence.git
cd Climora-Atmospheric-Intelligence
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r backend/requirements.txt
```

**4. Set environment variables:**

*Windows PowerShell:*
```powershell
$env:FLASK_SECRET_KEY="your_secret_key"
```

*Linux / macOS:*
```bash
export FLASK_SECRET_KEY="your_secret_key"
```

**5. Run the engine:**

*Windows:*
```cmd
cmd /c "set FLASK_SECRET_KEY=your_secret_key && cd backend && ..\.venv\Scripts\python.exe app.py"
```

*Linux / macOS:*
```bash
cd backend
python3 app.py
```

> The application will boot on **http://localhost:5000** using Waitress for production-ready WSGI serving.

---

## 🎯 Use Cases
- Meteorological research and data analysis
- Historical climate trend exploration
- Academic ML pipeline demonstrations
- Self-learning forecasting system experimentation

---

## 🔮 Future Advancements

Climora is designed as an evolving atmospheric intelligence platform.  
Future development focuses on deeper data fusion, real-time streams, and autonomous model evolution.

- **Live Weather Data Integration** – Real-time stream ingestion from public meteorological APIs
- **Multi-Variable Forecasting** – Extend beyond temperature to humidity, pressure, wind, and precipitation
- **Neural Architecture Upgrade** – LSTM or Transformer-based temporal sequence modeling
- **Ensemble Intelligence** – Weighted ensemble voting across multiple model families
- **AutoML Feedback** – Autonomous hyperparameter tuning based on historical MAE trends
- **Geospatial Awareness** – Location-tagged datasets with regional atmospheric pattern memory
- **Explainable Forecasts** – Feature attribution and natural language explanation of predictions
- **Multi-Dataset Comparison** – Side-by-side intelligence analysis across uploaded datasets

> *"Climora is not built to replace meteorologists,  
but to give them an intelligence layer that never stops learning."*

---

## 👨‍💻 Author

**Yash Kadam**  
AI & ML Engineer | Builder of Human-Centric, Intelligent Systems

> "I didn't want to build a weather app.  
> I wanted to build an engine that understands the sky."
