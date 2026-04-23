import os
import time
import hashlib
import json
from datetime import datetime, timedelta
import threading
import pandas as pd
import numpy as np
import joblib

from flask import render_template, request, flash, redirect, url_for, jsonify, g
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from utils.logger import logger
from utils.cache import load_feedback_history, save_feedback_history
from utils.metrics import training_lock, metrics_lock, system_metrics
from ml.hybrid_model import HybridWeatherModel
from ml.intelligence import (detect_regime_shifts, compute_uncertainty, 
                             compute_dynamic_confidence, classify_behavior, 
                             assess_risk, analyze_weather_context, apply_adaptive_feedback)
from visualization.plot import generate_forecast_plot

def register_routes(app):
    @app.route('/health')
    def health():
        return jsonify({"status": "healthy"})

    @app.route('/metrics')
    def metrics():
        with metrics_lock:
            reqs = system_metrics['total_requests']
            avg_time = system_metrics['total_response_time'] / reqs if reqs > 0 else 0
            return jsonify({
                "total_requests": reqs,
                "cache_hits": system_metrics['cache_hits'],
                "model_trainings": system_metrics['model_trainings'],
                "average_response_time_sec": round(avg_time, 4)
            })

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))

        if file and file.filename.endswith('.csv'):
            try:
                file_bytes = file.read()
                data_hash = hashlib.sha256(file_bytes).hexdigest()
                file.seek(0)
                df = pd.read_csv(file)
            except Exception as e:
                logger.error(f"Error reading CSV: {e}", extra={'event': 'csv_error'})
                flash(f"Error reading CSV: {e}")
                return redirect(url_for('index'))

            if 'Date' not in df.columns or 'Temperature' not in df.columns:
                flash("CSV must contain 'Date' and 'Temperature' columns.")
                return redirect(url_for('index'))

            if len(df) < 10:
                flash("CSV must contain at least 10 rows for reliable prediction.")
                return redirect(url_for('index'))

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
            if df.empty or len(df) < 10:
                 flash("Not enough valid date entries after parsing.")
                 return redirect(url_for('index'))

            df.sort_values('Date', inplace=True)
            df['Temperature'] = df['Temperature'].ffill().bfill()
            
            df['DayIndex'] = (df['Date'] - df['Date'].min()).dt.days
            df['Sin_Day'] = np.sin(2 * np.pi * df['Date'].dt.dayofyear / 365.25)
            df['Cos_Day'] = np.cos(2 * np.pi * df['Date'].dt.dayofyear / 365.25)
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            
            X = df[['DayIndex', 'Sin_Day', 'Cos_Day', 'Day', 'Month', 'DayOfWeek']]
            y = df['Temperature']
            
            context = analyze_weather_context(df, y)
            mean_temp = context["mean_temp"]
            std_dev = context["std_dev"]
            historical_trend = context["historical_trend"]
            variability = context["variability"]
            pattern = context["pattern"]
            anomalies = context["anomalies"]
            confidence = context["confidence"]
            is_unstable = context["is_unstable"]
            insight = context["insight"]
            
            if is_unstable:
                y_train_target = y.rolling(window=3, min_periods=1, center=True).mean()
                y_train_target = np.clip(y_train_target, mean_temp - 2*std_dev, mean_temp + 2*std_dev)
            else:
                y_train_target = y
            
            CACHE_DIR = app.config.get('CACHE_DIR', 'model_cache')
            MODEL_VERSION = app.config.get('MODEL_VERSION', '6.0')
            metadata_path = os.path.join(CACHE_DIR, f"{data_hash}_meta.json")
            model_path = os.path.join(CACHE_DIR, f"{data_hash}.joblib")
            
            use_cached_model = False
            if os.path.exists(metadata_path) and os.path.exists(model_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get('hash') == data_hash and metadata.get('model_version') == MODEL_VERSION:
                        use_cached_model = True
                except Exception as e:
                    logger.warning(f"Failed to read cache metadata: {e}", extra={'event': 'cache_read_error'})
                    
            if use_cached_model:
                logger.info("Cache hit: Loading model from disk.", extra={'event': 'cache_hit'})
                with metrics_lock:
                    system_metrics['cache_hits'] += 1
                try:
                    load_start = time.time()
                    model = joblib.load(model_path)
                    load_time = time.time() - load_start
                    logger.info("Model loaded successfully.", extra={'event': 'model_load', 'duration': round(load_time, 4)})
                    
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                except Exception as e:
                    logger.error(f"Failed to load cached model: {e}", extra={'event': 'model_load_error'})
                    use_cached_model = False
                    
            if not use_cached_model:
                with training_lock:
                    if os.path.exists(model_path) and os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            if metadata.get('hash') == data_hash and metadata.get('model_version') == MODEL_VERSION:
                                model = joblib.load(model_path)
                                y_pred = model.predict(X)
                                mse = mean_squared_error(y, y_pred)
                                r2 = r2_score(y, y_pred)
                                
                                use_cached_model = True
                                with metrics_lock:
                                    system_metrics['cache_hits'] += 1
                                logger.info("Cache hit (delayed): Model loaded via secondary check.", extra={'event': 'cache_hit_delayed'})
                        except Exception:
                            use_cached_model = False

                    if not use_cached_model:
                        logger.info("Cache miss: Retraining model synchronously.", extra={'event': 'cache_miss'})
                        with metrics_lock:
                            system_metrics['model_trainings'] += 1
                        train_start = time.time()
                        X_train, X_test, y_train, y_test = train_test_split(X, y_train_target, test_size=0.2, random_state=42)
                        
                        base_pipeline = make_pipeline(PolynomialFeatures(degree=1), RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
                        rf_pipeline = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                        model = HybridWeatherModel(base_pipeline, rf_pipeline)
                        model.fit(X_train, y_train)
                        
                        y_test_pred = model.predict(X_test)
                        test_mse = mean_squared_error(y_test, y_test_pred)
                        test_rmse = np.sqrt(test_mse)
                        test_mae = mean_absolute_error(y_test, y_test_pred)
                        test_r2 = r2_score(y_test, y_test_pred)
                        
                        logger.info(f"Model cross-validated. RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.2f}")
                        
                        model.fit(X, y_train_target)
                        train_time = time.time() - train_start
                        logger.info("Model trained from scratch.", extra={'event': 'model_train', 'duration': round(train_time, 4)})
                        
                        y_pred = model.predict(X)
                        mse = mean_squared_error(y, y_pred)
                        r2 = r2_score(y, y_pred)
                        
                        try:
                            temp_model_path = model_path + '.tmp'
                            joblib.dump(model, temp_model_path)
                            os.replace(temp_model_path, model_path)
                            
                            metadata = {
                                "hash": data_hash,
                                "model_version": MODEL_VERSION,
                                "timestamp": datetime.now().isoformat()
                            }
                            temp_metadata_path = metadata_path + '.tmp'
                            with open(temp_metadata_path, 'w') as f:
                                json.dump(metadata, f)
                            os.replace(temp_metadata_path, metadata_path)
                            logger.info("Model cached successfully.", extra={'event': 'model_cache_save'})
                        except Exception as e:
                            logger.error(f"Failed to cache model: {e}", extra={'event': 'model_cache_error'})

            def background_optimization_task(dataset_hash):
                logger.info(f"Background optimization triggered for {dataset_hash}", extra={'event': 'background_task_start'})
                time.sleep(1) 
                logger.info(f"Background optimization completed for {dataset_hash}", extra={'event': 'background_task_complete'})

            threading.Thread(target=background_optimization_task, args=(data_hash,), daemon=True).start()
            
            regime_shift_indices = detect_regime_shifts(y)
            prediction_margin = compute_uncertainty(y, y_pred, k=1.5)
            
            last_date = df['Date'].max()
            last_day_index = df['DayIndex'].max()
            
            future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
            
            future_features = []
            for i, f_date in enumerate(future_dates):
                f_day_index = last_day_index + i + 1
                f_sin = np.sin(2 * np.pi * f_date.dayofyear / 365.25)
                f_cos = np.cos(2 * np.pi * f_date.dayofyear / 365.25)
                f_day = f_date.day
                f_month = f_date.month
                f_dow = f_date.dayofweek
                future_features.append([f_day_index, f_sin, f_cos, f_day, f_month, f_dow])
                
            future_day_df = pd.DataFrame(future_features, columns=['DayIndex', 'Sin_Day', 'Cos_Day', 'Day', 'Month', 'DayOfWeek'])
            future_predictions = model.predict(future_day_df)
            
            confidence_per_day = compute_dynamic_confidence(confidence, prediction_margin / 1.5, len(future_predictions), decay_factor=1.5)
            prediction_behavior = classify_behavior(std_dev, len(regime_shift_indices), is_unstable, len(anomalies))
            risk_level = assess_risk(std_dev, len(anomalies), len(regime_shift_indices))
            
            prediction_intervals = [{"lower": round(p - prediction_margin, 2), "upper": round(p + prediction_margin, 2)} for p in future_predictions]
            
            feedback_results = apply_adaptive_feedback(y, y_pred, future_predictions, confidence_per_day, data_hash, CACHE_DIR, context)
            
            residual_history = feedback_results["residual_history"]
            bias_type = feedback_results["bias_type"]
            fb_mean_residual = feedback_results["fb_mean_residual"]
            fb_residual_std = feedback_results["fb_residual_std"]
            adjusted_forecast = feedback_results["adjusted_forecast"]
            bias_correction_applied = feedback_results["bias_correction_applied"]
            adjusted_confidence_per_day = feedback_results["adjusted_confidence_per_day"]
            improvement_score = feedback_results["improvement_score"]
            meta_insight = feedback_results["meta_insight"]
                
            avg_temp = np.mean(adjusted_forecast)
            min_temp = np.min(adjusted_forecast)
            max_temp = np.max(adjusted_forecast)
            
            if adjusted_forecast[-1] > adjusted_forecast[0] + 0.5:
                trend_label = "Increasing ↗"
            elif adjusted_forecast[-1] < adjusted_forecast[0] - 0.5:
                trend_label = "Decreasing ↘"
            else:
                trend_label = "Stable ➝"
            
            plot_url = generate_forecast_plot(df, y_pred, future_dates, adjusted_forecast)
            
            future_data = []
            for date, temp in zip(future_dates, adjusted_forecast):
                future_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'temp': round(temp, 2)
                })

            return render_template('result.html', 
                                   plot_url=plot_url, 
                                   mse=round(mse, 2), 
                                   r2=round(r2, 2),
                                   future_data=future_data,
                                   avg_temp=avg_temp,
                                   min_temp=min_temp,
                                   max_temp=max_temp,
                                   trend_label=trend_label,
                                   historical_trend=historical_trend,
                                   variability=variability,
                                   pattern=pattern,
                                   confidence=round(confidence, 1),
                                   insight_text=insight,
                                   anomaly_count=len(anomalies),
                                   regime_shift_indices=regime_shift_indices,
                                   prediction_intervals=prediction_intervals,
                                   confidence_per_day=confidence_per_day,
                                   prediction_behavior=prediction_behavior,
                                   risk_level=risk_level,
                                   residual_history=residual_history,
                                   bias_type=bias_type,
                                   mean_residual=round(fb_mean_residual, 2),
                                   residual_std=round(fb_residual_std, 2),
                                   adjusted_forecast=[round(p, 2) for p in adjusted_forecast],
                                   bias_correction_applied=bias_correction_applied,
                                   adjusted_confidence_per_day=adjusted_confidence_per_day,
                                   improvement_score=round(improvement_score, 1),
                                   meta_insight=meta_insight,
                                   dataset_hash=data_hash)
        else:
            flash('Invalid file format. Please upload a CSV.')
            return redirect(url_for('index'))
