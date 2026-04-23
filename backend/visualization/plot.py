import io
import base64
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def generate_forecast_plot(df, y_pred, future_dates, adjusted_forecast):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot historical data
    plt.plot(df['Date'], df['Temperature'], label='Historical Temperature', color='#3498db', linewidth=2)
    
    # Plot trend line (from regression model)
    plt.plot(df['Date'], y_pred, label='Trend Line', color='#e74c3c', linestyle='--', linewidth=2)
    
    # Plot future predictions
    plt.plot(future_dates, adjusted_forecast, label='7-Day Prediction (Adjusted)', color='#2ecc71', marker='o', linestyle='-', linewidth=2)
    
    plt.title('Weather Data Analysis & Temperature Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Temperature', fontsize=12)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to a base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    return plot_url
