import numpy as np
from sklearn.metrics import mean_absolute_error
from utils.cache import load_feedback_history, save_feedback_history

def detect_regime_shifts(y):
    # Detect changes in trend direction using slope differences
    diffs = np.diff(y)
    signs = np.sign(diffs)
    # Find indices where sign changes (ignoring 0)
    sign_changes = np.where(np.diff(signs) != 0)[0] + 1
    return sign_changes.tolist()

def compute_uncertainty(y_true, y_pred, k=1.5):
    # Compute residual standard deviation
    residuals = np.array(y_true) - np.array(y_pred)
    residual_std = np.std(residuals)
    return k * residual_std

def compute_dynamic_confidence(base_confidence, residual_std, forecast_days, decay_factor=1.0):
    # Generate confidence for each step
    confidences = []
    for i in range(forecast_days):
        # Base minus decay per day and standard deviation impact
        c = base_confidence - (decay_factor * (i + 1)) - (residual_std * 2)
        confidences.append(round(max(50.0, min(100.0, c)), 1))
    return confidences

def classify_behavior(std_dev, regime_shifts_count, is_unstable, anomalies_count):
    if std_dev > 5.0 or anomalies_count > 1 or regime_shifts_count > 5:
        return "High-Uncertainty Forecast"
    elif is_unstable:
        return "Smoothed Forecast"
    else:
        return "Stable Projection"

def assess_risk(std_dev, anomalies_count, regime_shifts_count):
    if std_dev > 5.0 or anomalies_count > 2 or regime_shifts_count > 5:
        return "High"
    elif std_dev > 2.0 or anomalies_count > 0 or regime_shifts_count > 2:
        return "Medium"
    else:
        return "Low"

def analyze_weather_context(df, y):
    mean_temp = np.mean(y)
    std_dev = np.std(y)
    if std_dev == 0.0:
        std_dev = 1e-5
        
    z = np.polyfit(df['DayIndex'], y, 1)
    hist_slope = z[0]
    if hist_slope > 0.05:
        historical_trend = "Increasing"
    elif hist_slope < -0.05:
        historical_trend = "Decreasing"
    else:
        historical_trend = "Stable"
        
    if std_dev < 2.0:
        variability = "Low"
    elif std_dev < 5.0:
        variability = "Moderate"
    else:
        variability = "High"
        
    if variability == "Low":
        pattern = "Stable pattern"
    elif abs(hist_slope) > 0.1 and variability != "High":
        pattern = "Linear trend"
    else:
        pattern = "Fluctuating pattern"
        
    z_scores = (y - mean_temp) / std_dev
    anomalies = np.where(np.abs(z_scores) > 2.5)[0]
    
    confidence = 100 - (std_dev * 5)
    confidence = max(50.0, min(100.0, confidence))
    
    diffs = np.diff(y)
    direction_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    is_unstable = (std_dev > 5.0) or (direction_changes > len(y) * 0.3)
    
    insight = f"Temperature shows a {'steady' if variability == 'Low' else 'fluctuating'} {historical_trend.lower()} trend with {variability.lower()} variability. "
    if is_unstable:
        insight += "Data exhibits instability; predictions have been smoothed for reliability. "
    if len(anomalies) > 0:
        insight += f"Detected {len(anomalies)} anomaly spike(s), forecast confidence is adjusted to {confidence:.1f}%."
    else:
        insight += f"No major anomalies detected. Forecast confidence is {confidence:.1f}%."
        
    return {
        "mean_temp": mean_temp,
        "std_dev": std_dev,
        "historical_trend": historical_trend,
        "variability": variability,
        "pattern": pattern,
        "anomalies": anomalies.tolist(),
        "confidence": confidence,
        "is_unstable": is_unstable,
        "insight": insight
    }

def apply_adaptive_feedback(y, y_pred, future_predictions, confidence_per_day, data_hash, cache_dir, context):
    feedback = load_feedback_history(data_hash, cache_dir)
    
    current_residuals = (np.array(y) - np.array(y_pred)).tolist()
    current_mae = float(mean_absolute_error(y, y_pred))
    
    feedback["residuals"].extend(current_residuals)
    feedback["residuals"] = feedback["residuals"][-200:]
    feedback["maes"].append(current_mae)
    feedback["maes"] = feedback["maes"][-20:]
    
    save_feedback_history(feedback, data_hash, cache_dir)
    
    residual_history = feedback["residuals"]
    fb_mean_residual = np.mean(residual_history) if len(residual_history) > 0 else 0.0
    fb_residual_std = np.std(residual_history) if len(residual_history) > 0 else 0.0
    
    threshold = 0.5
    if len(residual_history) >= 20:
        if fb_mean_residual > threshold:
            bias_type = "Underprediction"
        elif fb_mean_residual < -threshold:
            bias_type = "Overprediction"
        else:
            bias_type = "Neutral"
    else:
        bias_type = "Neutral"
        
    bias_correction_applied = False
    adjusted_forecast = np.array(future_predictions)
    
    if len(residual_history) >= 20 and bias_type != "Neutral" and fb_residual_std < 5.0:
        correction = np.clip(fb_mean_residual, -2.0, 2.0)
        adjusted_forecast = adjusted_forecast + correction
        bias_correction_applied = True
        
    # Stability Guard: Clip forecast to physical bounds based on historical data
    upper_bound = context["mean_temp"] + (3 * context["std_dev"])
    lower_bound = context["mean_temp"] - (3 * context["std_dev"])
    adjusted_forecast = np.clip(adjusted_forecast, lower_bound, upper_bound)
    
    adjusted_confidence_per_day = []
    alpha = 1.2
    for c in confidence_per_day:
        adj_c = c - (alpha * fb_residual_std)
        if fb_residual_std < 1.0:
            adj_c += 2.0
        adjusted_confidence_per_day.append(round(max(50.0, min(100.0, adj_c)), 1))
        
    # Enforce monotonic decay
    for i in range(1, len(adjusted_confidence_per_day)):
        if adjusted_confidence_per_day[i] > adjusted_confidence_per_day[i-1]:
            adjusted_confidence_per_day[i] = adjusted_confidence_per_day[i-1]
            
    maes = feedback["maes"]
    improvement_score = 0.0
    if len(maes) >= 4:
        mid = len(maes) // 2
        old_mae = np.mean(maes[:mid])
        new_mae = np.mean(maes[mid:])
        if old_mae > 0:
            improvement_score = ((old_mae - new_mae) / old_mae) * 100
            
    meta_insight = f"Model shows {bias_type.lower()} bias; "
    if bias_correction_applied:
        meta_insight += "forecasts adjusted accordingly. "
    else:
        meta_insight += "no correction applied. "
    
    if improvement_score > 5.0:
        meta_insight += "Prediction stability has improved over recent runs. "
    elif improvement_score < -5.0:
        meta_insight += "Prediction stability has degraded slightly. "
        
    if fb_residual_std > 3.0:
        meta_insight += "High residual variance detected; confidence reduced."
        
    return {
        "residual_history": residual_history,
        "bias_type": bias_type,
        "fb_mean_residual": fb_mean_residual,
        "fb_residual_std": fb_residual_std,
        "adjusted_forecast": adjusted_forecast.tolist(),
        "bias_correction_applied": bias_correction_applied,
        "adjusted_confidence_per_day": adjusted_confidence_per_day,
        "improvement_score": improvement_score,
        "meta_insight": meta_insight
    }
