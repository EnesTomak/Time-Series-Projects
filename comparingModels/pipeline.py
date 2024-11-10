import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv("data/tesla_sales_time_series.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index()
df = df.asfreq('MS')


# Random noise function
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


# Create date features
def create_date_features(df):
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['day_of_year'] = df.index.day_of_year
    df['month'] = df.index.month
    return df


# Lag features
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe[f'Sales_lag_{lag}'] = dataframe['Sales'].shift(lag)
    return dataframe


# Rolling mean features
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe[f'Sales_roll_mean_{window}'] = dataframe['Sales'].rolling(window=window).mean()
    return dataframe


# EWM features
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe[f'Sales_ewm_alpha_{str(alpha).replace(".", "")}_lag_{lag}'] = \
                dataframe['Sales'].shift(lag).ewm(alpha=alpha).mean()
    return dataframe


# Feature engineering steps
df['Sales_noisy'] = df['Sales'] + random_noise(df)
df = create_date_features(df)
df = lag_features(df, [1, 3, 6, 12])
df = roll_mean_features(df, [3, 6, 12])
df = ewm_features(df, [0.95, 0.9, 0.8], [1, 3, 6])

# Handle missing values
df.dropna(inplace=True)

# One-hot encode 'month' feature
df = pd.get_dummies(df, columns=['month'], drop_first=True)

# Split data into training and test sets
train = df.loc['2015-01-01':'2023-12-01']
test = df.loc['2024-01-01':'2024-06-01']


# Pipeline for TES and SARIMA (non-noisy data)
def tes_sarima_pipeline(train, test):
    # Holt-Winters (TES)
    best_alpha, best_beta, best_gamma = 0.2, 0.9, 0.3
    tes_model = ExponentialSmoothing(train['Sales'], trend="add", seasonal="add", seasonal_periods=12). \
        fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)
    y_pred_tes = tes_model.forecast(len(test))

    # SARIMA
    best_order = (0, 0, 1)
    best_seasonal_order = (0, 0, 1, 12)
    sarima_model = SARIMAX(train['Sales'], order=best_order, seasonal_order=best_seasonal_order).fit(disp=0)
    y_pred_sarima = sarima_model.get_forecast(steps=len(test)).predicted_mean

    return y_pred_tes, y_pred_sarima


# Pipeline for LightGBM (noisy data)
def lightgbm_pipeline(train, test):
    X_train = train.drop(columns=['Sales', 'Sales_noisy'], errors='ignore')
    X_test = test.drop(columns=['Sales', 'Sales_noisy'], errors='ignore')
    y_train = train['Sales_noisy']
    y_test = test['Sales']

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': 5,
        'verbose': 0,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'nthread': -1
    }

    model = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dvalid],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    y_pred_lgbm = model.predict(X_test)
    return y_pred_lgbm

# MAPE calculation function
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Run the pipelines
y_pred_tes, y_pred_sarima = tes_sarima_pipeline(train, test)
y_pred_lgbm = lightgbm_pipeline(train, test)

# Calculate MAPE for each model
mape_tes = mean_absolute_percentage_error(test['Sales'], y_pred_tes)
mape_sarima = mean_absolute_percentage_error(test['Sales'], y_pred_sarima)
mape_lgbm = mean_absolute_percentage_error(test['Sales'], y_pred_lgbm)

# Print MAPE values
print(f"TES Model MAPE: {mape_tes:.2f}%")
print(f"SARIMA Model MAPE: {mape_sarima:.2f}%")
print(f"LightGBM Model MAPE: {mape_lgbm:.2f}%")

# Combine results and visualize
results_df = pd.DataFrame({
    'Date': test.index,
    'Actual': test['Sales'],
    'TES_Predicted': y_pred_tes,
    'SARIMA_Predicted': y_pred_sarima,
    'LightGBM_Predicted': y_pred_lgbm
})

# Save results
results_df.to_csv("model_comparison_results_pipeline.csv", index=False)

# Plot the results
results_df = pd.read_csv("model_comparison_results_pipeline.csv", parse_dates=["Date"], index_col="Date")
plt.figure(figsize=(14, 8))
plt.plot(results_df.index, results_df['Actual'], label='Gerçek Test Seti', color='green')
plt.plot(results_df.index, results_df['TES_Predicted'], label='TES Tahmin Edilen', linestyle='--', color='blue')
plt.plot(results_df.index, results_df['SARIMA_Predicted'], label='SARIMA Tahmin Edilen', linestyle='--', color='red')
plt.plot(results_df.index, results_df['LightGBM_Predicted'], label='LightGBM Tahmin Edilen', linestyle='--', color='purple')

plt.xlabel('Tarih')
plt.ylabel('Satış')
plt.title('Modellerin Tahmin Sonuçlarının Karşılaştırılması')
plt.legend()
plt.grid(True)
plt.show()