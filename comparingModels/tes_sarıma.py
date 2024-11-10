import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import itertools
import warnings
warnings.filterwarnings('ignore')


# Veri Setini Yükleme
df = pd.read_csv("data/tesla_sales_time_series.csv", parse_dates=["Date"], index_col="Date")

# Tarihleri sıralama
df = df.sort_index()

# Eksik tarihleri doldurma (gerekiyorsa)
df = df.asfreq('MS')  # Ay başlangıcı olarak frekans belirleme

# Eğitim ve Test Setlerinin Oluşturulması
train = df.loc['2015-01-01':'2023-12-01']  # Eğitim seti 2015-2023
test = df.loc['2024-01-01':'2024-06-01']  # Test seti 2024'ün ilk 6 ayı

# MAPE hesaplama fonksiyonu
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# TES Optimizasyon Fonksiyonu
def tes_optimizer(train, test, abg, step=6):
    best_alpha, best_beta, best_gamma, best_mape = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_trend=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mape = mean_absolute_percentage_error(test, y_pred)
        if mape < best_mape:
            best_alpha, best_beta, best_gamma, best_mape = comb[0], comb[1], comb[2], mape
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mape, 2)])

    print("En İyi Parametreler - Alpha:", round(best_alpha, 2),
          "Beta:", round(best_beta, 2),
          "Gamma:", round(best_gamma, 2),
          "MAPE:", round(best_mape, 2), "%")

    return best_alpha, best_beta, best_gamma, best_mape

alpha_vals = beta_vals = gamma_vals = np.arange(0.1, 1.0, 0.10)
abg = list(itertools.product(alpha_vals, beta_vals, gamma_vals))
best_alpha, best_beta, best_gamma, best_mape = tes_optimizer(train['Sales'], test['Sales'], abg)

# Son TES Modeli (Optimizasyondan elde edilen parametrelerle)
final_tes_model = ExponentialSmoothing(train['Sales'], trend="add", seasonal="add", seasonal_periods=12). \
    fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)
y_pred_tes = final_tes_model.forecast(len(test))



# SARIMA Optimizasyon Fonksiyonu
def sarima_optimizer_mape(train, test, pdq, seasonal_pdq):
    best_mape, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                sarima_model = model.fit(disp=0)
                y_pred_test = sarima_model.get_forecast(steps=len(test))
                y_pred = y_pred_test.predicted_mean
                mape = mean_absolute_percentage_error(test, y_pred)
                if mape < best_mape:
                    best_mape, best_order, best_seasonal_order = mape, param, param_seasonal
                print('SARIMA{}x{}12 - MAPE:{}'.format(param, param_seasonal, mape))
            except Exception as e:
                print(f"Error for SARIMA{param}x{param_seasonal}: {e}")
                continue
    print('SARIMA{}x{}12 - MAPE:{}'.format(best_order, best_seasonal_order, best_mape))
    return best_order, best_seasonal_order

# Optimize search ranges with finer adjustments
pdq = [(p, d, q) for p in range(0, 2) for d in range(0, 2) for q in range(0, 2)]
seasonal_pdq = [(P, D, Q, S) for P in range(0, 2) for D in range(0, 2) for Q in range(0, 2) for S in [12]]
best_order, best_seasonal_order = sarima_optimizer_mape(train['Sales'], test['Sales'], pdq, seasonal_pdq)

# Son SARIMA Modeli (Optimizasyondan elde edilen parametrelerle)
model = SARIMAX(train['Sales'], order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)
y_pred_sarima = sarima_final_model.get_forecast(steps=len(test)).predicted_mean
y_pred_sarima = pd.Series(y_pred_sarima, index=test.index)

# Tahmin ve Gerçek Verileri Görselleştirme
plt.figure(figsize=(14, 8))
plt.plot(train.index, train['Sales'], label='Eğitim Seti')
plt.plot(test.index, test['Sales'], label='Gerçek Test Seti')
plt.plot(test.index, y_pred_tes, label='TES Tahmin Edilen', linestyle='--')
plt.plot(test.index, y_pred_sarima, label='SARIMA Tahmin Edilen', linestyle='--')
plt.legend()
plt.title('Holt-Winters ve SARIMA Modelleri Karşılaştırması')
plt.show()

# MAPE Hata Metriği Hesaplama
mape_tes = mean_absolute_percentage_error(test['Sales'], y_pred_tes)
mape_sarima = mean_absolute_percentage_error(test['Sales'], y_pred_sarima)

print(f'Holt-Winters MAPE: {mape_tes:.2f}%')
print(f'SARIMA MAPE: {mape_sarima:.2f}%')
