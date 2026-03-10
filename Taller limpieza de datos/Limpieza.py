# ============================================
# Taller: Limpieza y Preparación de Datos
# Librerías necesarias
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import unicodedata


# ============================================
# 1. Exploración inicial del dataset
# ============================================

# Cargar el dataset
df = pd.read_csv("datos_para_limpiar.csv")

# Mostrar primeras filas
print("Primeras 5 filas:")
print(df.head())

# Información general
print("\nInformación del DataFrame:")
print(df.info())

# Conteo de valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())


# ============================================
# 2. Eliminación de columnas innecesarias
# ============================================

df = df.drop(columns=["ID", "Notas"])


# ============================================
# 3. Normalización de nombres de columnas
# - minúsculas
# - sin espacios
# - sin tildes
# ============================================

def limpiar_nombre(col):
    col = col.lower().strip().replace(" ", "_")
    col = ''.join(
        c for c in unicodedata.normalize('NFD', col)
        if unicodedata.category(c) != 'Mn'
    )
    return col

df.columns = [limpiar_nombre(col) for col in df.columns]

print("\nColumnas normalizadas:")
print(df.columns)


# ============================================
# 4. Corrección de errores de escritura (ciudad)
# ============================================

df["ciudad"] = df["ciudad"].astype(str).str.strip().str.lower()

correccion_ciudades = {
    "bogota": "Bogotá",
    "bogotá": "Bogotá",
    "medellin": "Medellín",
    "medellín": "Medellín",
    "cali": "Cali"
}

df["ciudad"] = df["ciudad"].replace(correccion_ciudades)

# volver a convertir "nan" en NaN real
df["ciudad"] = df["ciudad"].replace("nan", np.nan)


# ============================================
# 5. Tratamiento de valores nulos
# ============================================

# Edad → reemplazar con media (solo edades válidas)
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

media_edades = df.loc[df["edad"].between(0,120), "edad"].mean()

df["edad"] = df["edad"].fillna(media_edades)

# Salario → reemplazar con mediana
df["salario"] = df["salario"].fillna(df["salario"].median())

# Ciudad y género → reemplazar con moda
df["ciudad"] = df["ciudad"].fillna(df["ciudad"].mode()[0])
df["genero"] = df["genero"].fillna(df["genero"].mode()[0])


# ============================================
# 6. Asegurar tipos de datos correctos
# ============================================

df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")


# ============================================
# 7. Filtrado de datos fuera de rango
# ============================================

# eliminar edades inválidas
df = df[df["edad"].between(0,120)]

# eliminar salarios extremos
df = df[df["salario"] <= 1_000_000_000]


# ============================================
# 8. Conversión de variables ordinales
# ============================================

mapa_nivel = {
    "bajo": 1,
    "medio": 2,
    "alto": 3
}

df["nivel"] = df["nivel"].str.lower().map(mapa_nivel)


# ============================================
# 9. Codificación de variables categóricas
# ============================================

df = pd.get_dummies(
    df,
    columns=["genero", "ciudad"],
    drop_first=True
)


# ============================================
# 10. Escalado de variables numéricas
# ============================================

scaler = MinMaxScaler()

df[["edad", "ingresos"]] = scaler.fit_transform(
    df[["edad", "ingresos"]]
)


# ============================================
# 11. Eliminación de registros duplicados
# ============================================

df = df.drop_duplicates()


# ============================================
# 12. Creación de nueva variable
# ============================================

df["utilidad"] = df["ingresos"] - df["gastos"]


# ============================================
# 13. Guardar dataset limpio
# ============================================

df.to_csv("datos_limpios.csv", index=False)

print("\nProceso completado. Archivo guardado como datos_limpios.csv")