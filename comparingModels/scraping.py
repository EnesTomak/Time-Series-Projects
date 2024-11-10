import requests
from bs4 import BeautifulSoup
import pandas as pd

# Web sayfası URL'si
url = 'https://www.goodcarbadcar.net/tesla-us-sales-figures/'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Tabloyu bul
table = soup.find('table', id='table_1')

# Başlıkları ayıkla
headers = [header.text.strip() for header in table.find_all('th') if header.text.strip()]

# Satırları ayıkla
rows = table.find_all('tr')
data = []

for row in rows[1:]:
    cols = row.find_all('td')
    cols = [ele.text.strip().replace(',', '') for ele in cols]
    if len(cols) == 13:  # Beklenen Yıl + 12 Ay
        data.append(cols)

df = pd.DataFrame(data, columns=headers)

# Pandas yapısını geniş formattan uzun formata yeniden şekillendir
df = pd.melt(df, id_vars=["Year"], var_name="Month", value_name="Sales")

# 'Ay'ı sayıya dönüştür
months = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}
df['Month'] = df['Month'].map(months)

# Bir 'Tarih' sütunu oluşturun ve bunu dizin olarak ayarla
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'])
df.set_index('Date', inplace=True)

# Artık gerekli olmadıkları için 'Yıl' ve 'Ay' sütunlarını bırak
df.drop(columns=['Year', 'Month'], inplace=True)

# 'Satışlar' sütununu sayısal değere dönüştürerek hataları NaN olarak zorlar
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')

# sıfır satışları NaN ile değiştir
df['Sales'] = df['Sales'].replace(0, pd.NA)

# İleri ve geri doldurma kullanarak eksik değerleri doldurun
df = df.bfill().ffill()

# CSV'ye kaydet
df.to_csv('tesla_sales_time_series.csv')

print(" Veriler başarılı bir şekilde kazındı, dönüştürüldü ve 'tesla_sales_time_series.csv'.")
