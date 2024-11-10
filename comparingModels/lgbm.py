import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

######### 1. Veri Yükleme ve İlk İşlemler #########
# Veri Yükleme
df = pd.read_csv("data/tesla_sales_time_series.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index()
df = df.asfreq('MS')

# İlk veriye bakış
print(df.head())
print(df.info())

######### 2. Feature Engineering #########

def create_date_features(df):
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['day_of_year'] = df.index.day_of_year
    df['month'] = df.index.month

    return df

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe[f'Sales_lag_{lag}'] = dataframe['Sales'].shift(lag)
    return dataframe

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe[f'Sales_roll_mean_{window}'] = dataframe['Sales'].rolling(window=window).mean()
    return dataframe

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe[f'Sales_ewm_alpha_{str(alpha).replace(".", "")}_lag_{lag}'] = \
                dataframe['Sales'].shift(lag).ewm(alpha=alpha).mean()
    return dataframe

df = create_date_features(df)

# Önem derecesine göre özellikler oluşturun
df = lag_features(df, [1, 3, 6, 12])
df = roll_mean_features(df, [3, 6, 12])
df = ewm_features(df, [0.95, 0.9, 0.8], [1, 3, 6])

# Eksik Değerleri İşleme
df.dropna(inplace=True)

df = pd.get_dummies(df, columns=['month'], drop_first=True)

# Güncellenmiş DataFrame'i görüntüle
print(df.head())

######### 3. Rastgele Gürültü Ekleme #########

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

# Rastgele gürültü uygulama
df['Sales_noisy'] = df['Sales'] + random_noise(df)


#########  4. Model Eğitimi ve Zaman Serisi Doğrulama Seti  #########

# Train-Test Ayrımı
train = df.loc['2015-01-01':'2023-12-01']
test = df.loc['2024-01-01':'2024-06-01']

# Özellik ve Hedef Ayrımı
X_train = train.drop(columns=['Sales', 'Sales_noisy'])
y_train = train['Sales_noisy']  # Use noisy sales for training
X_test = test.drop(columns=['Sales', 'Sales_noisy'])
y_test = test['Sales']  # Use actual sales for testing

# LightGBM Dataset Oluşturma
dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

# Model Parametreleri
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

# Modeli Eğit
model = lgb.train(
    params=params,
    train_set=dtrain,
    valid_sets=[dvalid],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=0),
        lgb.log_evaluation(period=100)
    ]
)

# Tahmin
y_pred_test = model.predict(X_test)

# MAPE Hesaplama
def mape(preds, actuals):
    return 100 * np.mean(2 * np.abs(preds - actuals) / (np.abs(preds) + np.abs(actuals)))

mape_value = mape(y_pred_test, y_test)
print("Test MAPE: ", mape_value)

# Tahmin ve Gerçek Verileri Görselleştirme
plt.figure(figsize=(14, 8))
plt.plot(train.index, train['Sales_noisy'], label='Eğitim Seti (Noisy)', color='blue')
plt.plot(test.index, test['Sales'], label='Gerçek Test Seti', color='green')
plt.plot(test.index, y_pred_test, label='LightGBM Tahmin Edilen', linestyle='--', color='red')
plt.legend()
plt.title('LightGBM Modeli ile Tesla Satış Tahminleri')
plt.xlabel('Tarih')
plt.ylabel('Satış')
plt.grid(True)
plt.show()


######### 5. Değişken Önem Düzeylerini Görselleştirme #########

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp


feat_imp = plot_lgb_importances(model, num=200)
plot_lgb_importances(model, num=30, plot=True)

# Önemi sıfır olan özellikleri filtreleme
importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values
cols = X_train.columns
imp_feats = [col for col in cols if col not in importance_zero]
print(f"Number of important features: {len(imp_feats)}")
