class HybridWeatherModel:
    def __init__(self, base_model, residual_model):
        self.base_model = base_model
        self.residual_model = residual_model
        
    def fit(self, X, y):
        # Fit base model on just the trend component (DayIndex)
        self.base_model.fit(X[['DayIndex']], y)
        base_preds = self.base_model.predict(X[['DayIndex']])
        residuals = y - base_preds
        # Fit residual model on all features
        self.residual_model.fit(X, residuals)
        return self
        
    def predict(self, X):
        base_preds = self.base_model.predict(X[['DayIndex']])
        res_preds = self.residual_model.predict(X)
        return base_preds + res_preds
