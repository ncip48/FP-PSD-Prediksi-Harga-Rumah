import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

def clean_data(df):
    def parse_price(x):
        x = str(x).lower()
        num = float(re.sub(r'[^0-9,]', '', x).replace(',', '.'))
        if 'miliar' in x:
            return num * 1000
        elif 'juta' in x:
            return num
        return num
        
    df['price_million'] = df['price'].apply(parse_price)
    df['log_price_million'] = np.log(df['price_million'])
    return df

def train_evaluate_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return model, mae, rmse, r2

def check_assumptions(X, y):
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    
    # VIF
    vif = pd.DataFrame()
    vif['Variable'] = X_sm.columns
    vif['VIF'] = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
    
    # Heteroskedasticity
    bp_test = het_breuschpagan(model_sm.resid, X_sm)
    
    # Normality
    shapiro_test = shapiro(model_sm.resid)
    
    return vif, bp_test, shapiro_test, model_sm.resid