from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from .forms import CriptoForm

import os
import io
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
import base64

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def main(request):
    imagen_grafico = None
    if request.method == 'POST':
        form = CriptoForm(request.POST)
        if form.is_valid():
            cryptocurrency = form.cleaned_data['criptomoneda']
            date = form.cleaned_data['fecha']

            model_path = os.path.join(settings.BASE_DIR, 'CryptoApp', 'models', 'model.pkl')
            scaler_path = os.path.join(settings.BASE_DIR, 'CryptoApp', 'models', 'scaler.pkl')
            data_path = os.path.join(settings.BASE_DIR, 'CryptoApp', 'models', 'modeling_data.csv')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            df = pd.read_csv(data_path, sep=';', encoding='utf-8')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df.asfreq('h')
            # CRIPTOMONEDA DATOS
            cryptocurrency = 'xrp'
            df_data = df[[f'price_{cryptocurrency}', f'volume_{cryptocurrency}']].iloc[-8760:]
            df_price = pd.DataFrame(df_data[f'price_{cryptocurrency}'])
            df_price.columns = ['price']
            data = df_data.values
            normalized_data = scaler.transform(data)
            input = np.expand_dims(normalized_data, axis=0)
            predictions = model.predict(input, verbose=0)
            future_dates = pd.date_range(start=df_price.index[-1] + pd.Timedelta(hours=1), periods=720, freq='h')
            df_predictions = pd.DataFrame({
                'price': predictions[0]
            }, index=future_dates)
            max_prediction = df_predictions['price'].max()
            first_max_index = df_predictions[df_predictions['price'] == max_prediction].index[0]
            plt.figure(figsize=(10, 6), facecolor='black')
            plt.axes().set_facecolor('black')
            plt.plot(df.index, df[f'price_{cryptocurrency}'], color='blue', linewidth=1, label='Datos Originales')
            plt.plot(df_predictions.index, df_predictions['price'], color='red', linewidth=1, label='Predicciones')
            plt.axhline(y=df_predictions['price'].max(), color='orange', linewidth=0.5, label='Máximo Predicciones')
            plt.axhline(y=df_price['price'][-1], color='green', linewidth=0.5, label='Máximo Predicciones')
            plt.axvline(x=first_max_index, color='orange', linewidth=0.5, label='Primera Coincidencia Máximo')
            plt.axvline(x=df_price.index[-1], color='green', linewidth=0.5, label='Primera Coincidencia Máximo')
            plt.tick_params(axis='x', colors='white')  # Color de las etiquetas del eje X
            plt.tick_params(axis='y', colors='white')  # Color de las etiquetas del eje Y
            plt.grid(alpha=0.3, linestyle='--', color='gray')  # Ejemplo de grid con transparencia y estilo punteado
            plt.tight_layout()
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            #buffer = io.BytesIO()
            #plt.savefig(buffer, format='png')
            #buffer.seek(0)

            # Convertir el gráfico a formato base64 para incrustarlo en la plantilla
            #image_png = buffer.getvalue()
            #buffer.close()
            #graphic = urllib.parse.quote(image_png)
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            image_png = buffer.getvalue()
            buffer.close()
            graphic = base64.b64encode(image_png).decode('utf-8')

            # Renderizar la plantilla con el gráfico generado
            return render(request, 'main.html', {'imagen_grafico': 'data:image/png;base64,' + graphic})

            # Renderizar la plantilla con el gráfico generado
            #return render(request, 'main.html', {'imagen_grafico': 'data:image/png;base64,' + graphic})

            #plt.savefig('static/images/grafico.png', dpi=300, bbox_inches='tight', pad_inches=0)

            #plt.show()
            #return HttpResponse(f"Seleccionaste {cryptocurrency} en la fecha {date}.")
    else:
        form = CriptoForm()
    context = {
        'form': form,
        'imagen_grafico': imagen_grafico,
    }


    return render(request, 'main.html', context)
