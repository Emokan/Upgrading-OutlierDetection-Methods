# https://www.kaggle.com/code/gauravduttakiit/local-outlier-factor

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pycaret.anomaly import *

# Veri setini oku
data = pd.read_csv('/Users/dogu/Desktop/graduated/lofTutorial/nyc_taxi.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Başlangıç verisini kontrol et
data.head()

# Hareketli ortalamalar oluştur
data['MA60'] = data['value'].rolling(60).mean()
data['MA365'] = data['value'].rolling(365).mean()

# Hareketli ortalama sütunlarıyla son veriyi kontrol et
data.tail()

# Veriyi görselleştir
fig = px.line(data, x="timestamp", y=['value', 'MA60', 'MA365'], title='NYC Taxi Trips', template='plotly_dark')
fig.show()

# Hareketli ortalama sütunlarını sil
data.drop(['MA60', 'MA365'], axis=1, inplace=True)

# Timestamp'i indeks olarak ayarla
data.set_index('timestamp', drop=True, inplace=True)

# Zaman serisini saatlik olarak yeniden örnekle
data = data.resample('H').sum()

# Zaman bilgisinden yeni özellikler oluştur
data['day'] = [i.day for i in data.index]
data['day_name'] = [i.day_name() for i in data.index]
data['day_of_year'] = [i.dayofyear for i in data.index]
data['week_of_year'] = [i.weekofyear for i in data.index]
data['hour'] = [i.hour for i in data.index]
data['is_weekday'] = [i.isoweekday() for i in data.index]

# PyCaret setup
s = setup(data, session_id=42,
          ordinal_features={'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday', 'Saturday']},
          numeric_features=['is_weekday'])

# Mevcut modelleri kontrol et
models()

# LOF modelini oluştur
lof = create_model('lof')

# Modeli eğit ve sonuçları al
lof_results = assign_model(lof)

# Aykırı değerleri kontrol et
lof_results[lof_results['Anomaly'] == 1].head()

# Veriyi görselleştir
fig = px.line(lof_results, x=lof_results.index, y="value", title='NYC TAXI TRIPS - UNSUPERVISED ANOMALY DETECTION', template='plotly_dark')

# Aykırı değerlerin tarihlerini al
outlier_dates = lof_results[lof_results['Anomaly'] == 1].index

# Aykırı değerlerin y değerlerini al
y_values = [lof_results.loc[i]['value'] for i in outlier_dates]

# Aykırı değerleri kırmızı nokta olarak ekle
fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode='markers',
                        name='Anomaly', 
                        marker=dict(color='red', size=5)))

# Grafiği göster
fig.show()

# Modeli görselleştir
plot_model(lof)

# UMAP görselleştirmesi oluştur
plot_model(lof, plot='umap')