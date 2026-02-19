
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Casos_positivos_de_COVID-19_en_Colombia.csv", low_memory=False)

df['Fecha de reporte web'] = pd.to_datetime(df['Fecha de reporte web'], errors='coerce')


print("Dimensiones del dataset:", df.shape)
print("\nColumnas:")
print(df.columns)

casos_departamento = df['Departamento'].value_counts()

print("\nCasos por departamento:")
print(casos_departamento)

casos_departamento.plot(kind='bar', figsize=(12,6))
plt.title("Casos Totales por Departamento")
plt.xlabel("Departamento")
plt.ylabel("Número de Casos")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(8,5))
df['Edad'].plot(kind='hist', bins=20)
plt.title("Distribución de Edades de Casos COVID")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()


plt.figure(figsize=(8,6))
df.boxplot(column='Edad', by='Estado')
plt.title("Edad según Estado Clínico")
plt.suptitle("")  
plt.xlabel("Estado")
plt.ylabel("Edad")
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(df['Fecha de reporte web'], df['Edad'])
plt.title("Edad vs Fecha de Reporte")
plt.xlabel("Fecha de Reporte")
plt.ylabel("Edad")
plt.show()

casos_por_fecha = df.groupby('Fecha de reporte web').size()

plt.figure(figsize=(12,6))
plt.plot(casos_por_fecha.index, casos_por_fecha.values)
plt.title("Evolución de Casos COVID en el Tiempo")
plt.xlabel("Fecha")
plt.ylabel("Número de Casos")
plt.show()

print("\nTaller completado correctamente.")