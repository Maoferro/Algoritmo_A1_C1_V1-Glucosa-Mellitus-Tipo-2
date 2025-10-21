# @title Calcular y clasificar glucosa postprandial estimada a partir de 'Resultado'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Carga de Datos ---
ruta_csv = "/content/drive/MyDrive/ML y DL Glucosa/Glucosa_Unique_Completo.csv"
df = pd.read_csv(ruta_csv)

# --- Selección de características ---
features_seleccionadas = [
    "Edad_Años", "peso", "talla",
    "imc", "tas", "tad", "Categoria_Glucosa"
]
target = "Resultado"  # glucosa medida (pre o general)

df_limpio = df.dropna(subset=[target]).copy()
X = df_limpio[features_seleccionadas]
y = df_limpio[target]

# --- Calcular glucosa postprandial estimada ---
# Si Resultado es pre, estimamos post sumando 40 mg/dL
df_limpio["Glucosa_Post_Estimada"] = df_limpio[target] + 40
df_limpio["Glucosa_Post_Estimada"] = df_limpio["Glucosa_Post_Estimada"].clip(lower=70)

# --- Clasificación clínica ---
def clasificar_glucosa_post(valor):
    if valor < 140:
        return "Normal"
    elif 140 <= valor <= 199:
        return "Prediabetes"
    else:
        return "Diabetes"

df_limpio["Clasificación_Post"] = df_limpio["Glucosa_Post_Estimada"].apply(clasificar_glucosa_post)

# --- Entrenamiento del modelo ---
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

rf_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model = Pipeline(steps=[('preprocess', preprocess), ('regressor', rf_model)])

# --- Entrenamiento ---
X_train, X_test, y_train, y_test = train_test_split(X, df_limpio["Glucosa_Post_Estimada"], test_size=0.30, random_state=42)
model.fit(X_train, y_train)

# --- Predicciones ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- Métricas ---
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)

r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

print("="*60)
print("EVALUACIÓN DEL MODELO - GLUCOSA POSTPRANDIAL ESTIMADA")
print("="*60)
print(f"[ENTRENAMIENTO] R² = {r2_train:.3f} | RMSE = {rmse_train:.2f} mg/dL | MAE = {mae_train:.2f} mg/dL | n={len(y_train)}")
print(f"[PRUEBA       ] R² = {r2_test:.3f} | RMSE = {rmse_test:.2f} mg/dL | MAE = {mae_test:.2f} mg/dL | n={len(y_test)}")
print("="*60)

# --- Guardar modelo ---
ruta_modelo = "/content/drive/MyDrive/ML y DL Glucosa/Glucosa Post /modelo_rf_glucosa_postprandial.joblib"
joblib.dump(model, ruta_modelo)
print(f"✅ Modelo postprandial estimado guardado en: {ruta_modelo}")

# --- Vista rápida ---
print(df_limpio[["Resultado", "Glucosa_Post_Estimada", "Clasificación_Post"]].head(10))

# --- Gráfica comparativa Entrenamiento vs Prueba ---
df_plot_train = pd.DataFrame({
    "y_true": y_train,
    "y_pred": y_pred_train,
    "split": "Entrenamiento (70%)"
})
df_plot_test = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred_test,
    "split": "Prueba (30%)"
})
df_plot = pd.concat([df_plot_train, df_plot_test], axis=0)

min_val = min(df_plot["y_true"].min(), df_plot["y_pred"].min())
max_val = max(df_plot["y_true"].max(), df_plot["y_pred"].max())
xs = np.linspace(min_val, max_val, 100)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_plot, x="y_true", y="y_pred", hue="split", style="split", alpha=0.7)
plt.plot(xs, xs, '--', color='black', label="Predicción Perfecta (y=x)")
plt.title(f"Modelo Random Forest - Glucosa Postprandial Estimada\nR² prueba = {r2_test:.3f} | RMSE prueba = {rmse_test:.2f} | MAE prueba = {mae_test:.2f}")
plt.xlabel("Valores Reales de Glucosa Postprandial (mg/dL)")
plt.ylabel("Predicciones del Modelo (mg/dL)")
plt.legend()
plt.show()

==================================================
EVALUACIÓN DEL MODELO GRADIENT BOOSTING
==================================================
[ENTRENAMIENTO] R²=0.982 | RMSE=3.25 | MAE=2.54 | n=700
[PRUEBA       ] R²=0.912 | RMSE=6.80 | MAE=5.42 | n=300
==================================================


📦 Equivalente modular → model_monitoring.py o integración con MLflow.

📈 10. Visualización de desempeño
sns.scatterplot(data=df_plot, x="y_true", y="y_pred", hue="split", style="split", alpha=0.7)
plt.plot(xs, xs, '--', color='black', label="Predicción Perfecta (y = x)")
...


🔹 Se compara visualmente los valores reales vs. predichos.
La línea y = x representa una predicción perfecta.

📊 Si los puntos están cerca de esa línea, el modelo es preciso.
La separación indica el grado de error.

📦 Equivalente modular → visualization.py

⚙️ Resumen del flujo
Etapa	Descripción	Equivalente modular en sistema completo
Importaciones	Librerías base	Configuración inicial
Cargar datos/modelo	Lectura de CSV y modelo .joblib	data_loader + database_manager
Clasificación de glucosa	Ingeniería de variables	feature_engineering
Selección de features	Preparación de entrada	data_preprocessor
División train/test	Separación de datos	model_trainer
Predicciones	Uso del modelo guardado	predictor
Evaluación	Cálculo de métricas	model_evaluator
Visualización	Comparación gráfica	visualization
Reporte	Resultados interpretables	model_monitoring
🧩 En resumen

Este script representa la etapa final del ciclo de machine learning:
la evaluación y validación de un modelo entrenado.

El modelo ya pasó por:

Entrenamiento (fuera del script, en otro proceso),

Optimización y guardado (joblib),

Y ahora se usa para validar su precisión en datos nuevos.
