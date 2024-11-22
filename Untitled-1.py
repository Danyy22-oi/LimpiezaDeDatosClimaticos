import pandas as pd
from sklearn.preprocessing import MinMaxScaler # type: ignore

# Cargar los datos CSV 
df = pd.read_csv('clima.csv')

# 1. Verificar las primeras filas y la estructura de los datos
print("Primeras filas del dataset:")
print(df.head())

print("\nInformación del dataset:")
print(df.info())

# 2. Manejo de valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# rellenamos valores nulos  con la media de cada columna numérica
columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
for col in columnas_numericas:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# 3. Eliminar duplicados 
df = df.drop_duplicates()

# 4. Convertir formatos
df['Año'] = df['Año'].astype(int)

# 5. Manejo de valores atípicos en las columnas numéricas
for col in columnas_numericas:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    # Filtrar valores dentro del rango
    df = df[(df[col] >= limite_inferior) & (df[col] <= limite_superior)]

# 6. Normalización de columnas seleccionadas
scaler = MinMaxScaler()
columnas_a_normalizar = ['Temperatura Promedio (°C)', 'Precipitación Promedio (mm)', 
                         'Humedad Promedio (%)', 'Velocidad del Viento Promedio (km/h)', 
                         'Horas de Sol Promedio (horas)', 'Índice UV Promedio', 
                         'Presión Atmosférica Promedio (hPa)', 'Días de Lluvia Promedio (días)']

df[columnas_a_normalizar] = scaler.fit_transform(df[columnas_a_normalizar])

# 7. Guardar los datos limpios
df.to_csv('datos_climaticos_limpios.csv', index=False)

print("\nLimpieza completada. Los datos limpios se han guardado en 'datos_climaticos_limpios.csv'.")
