# 🩺 Predicción de Glucosa con Machine Learning

## 📋 Descripción General

Este proyecto desarrolla y compara **7 modelos de Machine Learning** diferentes para predecir niveles de glucosa en sangre basándose en características clínicas y antropométricas del paciente. Todos los modelos siguen la misma metodología y pipeline, variando solo el algoritmo de entrenamiento.

Los modelos entrenados son:
1. Regresión Lineal
2. Ridge Regression
3. Lasso Regression
4. Random Forest
5. **Gradient Boosting** (usado como ejemplo en esta documentación)
6. Support Vector Machine (SVM)
7. Red Neuronal (MLP)

---

## 🎯 Objetivo

- Desarrollar múltiples modelos de predicción de glucosa
- Comparar desempeño entre diferentes algoritmos
- Identificar el modelo óptimo para máxima precisión
- Proporcionar una estructura reutilizable para cada modelo
- Clasificar automáticamente resultados según criterios clínicos

---

## 📦 Dependencias Requeridas

### Instalación

```bash
pip install pandas numpy scikit-learn joblib
```

### Versiones Recomendadas

```bash
pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 joblib>=1.0.0
```

---

## 🔧 Librerías Utilizadas

### **Pandas**
```python
import pandas as pd
```

| Función | Descripción |
|---------|-------------|
| `pd.read_csv()` | Carga archivo CSV desde Google Drive |
| `df.dropna()` | Elimina filas con valores faltantes |
| `df.apply()` | Aplica función a cada fila para crear categorías |

**Uso en este proyecto**: Manipulación y limpieza de datos tabulares

---

### **NumPy**
```python
import numpy as np
```

- **Función**: Operaciones numéricas subyacentes
- **Uso**: Cálculos de métricas (RMSE, MAE)
- **Por qué**: Base computacional eficiente para scikit-learn

---

### **Scikit-Learn**

#### `train_test_split`
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

- **Función**: Divide datos en 70% entrenamiento y 30% prueba
- **random_state=42**: Garantiza reproducibilidad
- **Ventaja**: Evalúa el modelo en datos nunca vistos

---

#### `OneHotEncoder`
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
```

- **Función**: Convierte variable categórica en numéricas binarias
- **Ejemplo**:
  ```
  Entrada:  ["Normal", "Prediabetes", "Diabetes"]
  Salida:   [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  ```
- **handle_unknown="ignore"**: Maneja categorías no vistas en entrenamiento

---

#### `ColumnTransformer`
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)
```

- **Función**: Aplica transformaciones diferentes por tipo de columna
- **"passthrough"**: Deja columnas numéricas sin cambios
- **OneHotEncoder**: Transforma solo las categóricas
- **Ventaja**: Preprocesamiento heterogéneo en una línea

---

#### `Pipeline`
```python
from sklearn.pipeline import Pipeline

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(...))
])
```

- **Función**: Encadena preprocesamiento y modelo
- **Ventaja**: Evita data leakage (fuga de información entre train/test)
- **Flujo**: `Datos Brutos → Preprocessor → Modelo → Predicción`

---

#### `Métricas de Evaluación`
```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

r2 = r2_score(y_real, y_predicho)
rmse = np.sqrt(mean_squared_error(y_real, y_predicho))
mae = mean_absolute_error(y_real, y_predicho)
```

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **R²** | `1 - (SS_res / SS_tot)` | Proporción de varianza explicada (0-1) |
| **RMSE** | `√(Σ(y_real - y_pred)² / n)` | Error promedio en mg/dL |
| **MAE** | `Σ\|y_real - y_pred\| / n` | Error absoluto medio en mg/dL |

---

### **Joblib**
```python
import joblib

# Guardar modelo
joblib.dump(model, "modelo.joblib")

# Cargar modelo
modelo_cargado = joblib.load("modelo.joblib")
```

- **Función**: Serializa modelos entrenados
- **Ventaja**: Reutilizar sin reentrenamiento

---

## 📊 Características (Features)

El modelo utiliza **7 características** como entrada:

| Feature | Tipo | Descripción | Rango Típico |
|---------|------|-------------|--------------|
| **Edad_Años** | Numérica | Edad del paciente | 18-100 años |
| **peso** | Numérica | Peso corporal | kg |
| **talla** | Numérica | Altura/Talla | cm |
| **imc** | Numérica | Índice de Masa Corporal | 10-50 kg/m² |
| **tas** | Numérica | Tensión Arterial Sistólica | 80-200 mmHg |
| **tad** | Numérica | Tensión Arterial Diastólica | 40-120 mmHg |
| **Categoria_Glucosa** | Categórica | Clasificación previa de glucosa | Nominal |

### Variable Objetivo (Target)
- **"Resultado"**: Medición de glucosa en sangre (mg/dL)

---

## 🏷️ Clasificación Clínica Automática

```python
def clasificar_glucosa(valor):
    if valor <= 100:
        return "Normal"
    elif valor <= 125:
        return "Prediabetes"
    else:
        return "Diabetes"
```

### Criterios de Clasificación (OMS/ADA)

| Clasificación | Rango (mg/dL) | Interpretación |
|---------------|---------------|----------------|
| **Normal** | ≤ 100 | Glucosa normal en ayunas |
| **Prediabetes** | 101-125 | Riesgo de desarrollar diabetes |
| **Diabetes** | ≥ 126 | Diabetes mellitus diagnosticada |

---

## 🔄 Estructura Común del Código

Todos los 7 modelos siguen esta estructura idéntica:

```
┌─────────────────────────────────────┐
│  1. CARGAR DATOS (CSV)              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. CREAR CATEGORÍAS DE GLUCOSA     │
│     (Normal/Prediabetes/Diabetes)   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. SELECCIONAR FEATURES (7)        │
│     y TARGET (Resultado)            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. LIMPIAR DATOS                   │
│     dropna() en target              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. PREPROCESAR                     │
│     Numéricas: passthrough          │
│     Categóricas: OneHotEncoder      │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  6. DIVIDIR DATOS                   │
│     70% train / 30% test            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  7. CREAR PIPELINE                  │
│     Preprocessor + Modelo           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  8. ENTRENAR MODELO                 │
│     model.fit(X_train, y_train)     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  9. EVALUAR                         │
│     Calcular R², RMSE, MAE          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  10. GUARDAR MODELO                 │
│      joblib.dump()                  │
└─────────────────────────────────────┘
```

---

## 📈 Comparativa de Desempeño

Se entrenaron 7 modelos diferentes. Aquí está el ranking:

| Modelo | R² Test | RMSE Test | MAE Test |
|--------|---------|-----------|----------|
| **Random Forest** 🥇 | 0.8622 | 12.62 mg/dL | 9.69 mg/dL |
| **Gradient Boosting** 🥈 | 0.8240 | 14.27 mg/dL | 10.90 mg/dL |
| **Ridge Regression** 🥉 | 0.8237 | 14.28 mg/dL | 11.26 mg/dL |
| Lasso Regression | 0.8237 | 14.28 mg/dL | 11.24 mg/dL |
| Regresión Lineal | 0.8233 | 14.29 mg/dL | 11.28 mg/dL |
| Support Vector Machine | 0.8114 | 14.77 mg/dL | 11.55 mg/dL |
| Red Neuronal (MLP) | 0.7956 | 15.37 mg/dL | 11.94 mg/dL |

---

## 🎯 Modelos Implementados

### 1. Regresión Lineal
- **Características**: Simple, interpretable, baseline
- **Ventaja**: Muy rápida
- **Desventaja**: No captura no-linealidades
- **Caso de uso**: Comparación base

### 2. Ridge Regression (L2 Regularization)
- **Características**: Regresión lineal con penalización
- **Ventaja**: Evita sobreajuste
- **Desventaja**: Mantiene todas las variables
- **Caso de uso**: Cuando se necesita estabilidad

### 3. Lasso Regression (L1 Regularization)
- **Características**: Regresión lineal con sparsity
- **Ventaja**: Selecciona automáticamente features importantes
- **Desventaja**: Puede eliminar variables útiles
- **Caso de uso**: Selección de variables

### 4. Random Forest
- **Características**: Ensamble de árboles paralelos (RECOMENDADO)
- **Ventaja**: Mejor precisión (R²=0.8622), robusto
- **Desventaja**: Requiere más memoria
- **Caso de uso**: Máxima precisión

### 5. Gradient Boosting
- **Características**: Ensamble secuencial de árboles débiles
- **Ventaja**: Muy buena precisión (R²=0.8240)
- **Desventaja**: Lento en entrenamiento
- **Caso de uso**: Datos complejos con relaciones no-lineales

### 6. Support Vector Machine (SVM)
- **Características**: Busca hiperplano óptimo
- **Ventaja**: Bueno en espacios de alta dimensión
- **Desventaja**: R² menor (0.8114)
- **Caso de uso**: Cuando hay muchas features

### 7. Red Neuronal (MLP)
- **Características**: Redes con capas densas
- **Ventaja**: Flexible, aprende patrones complejos
- **Desventaja**: Desempeño menor (R²=0.7956), requiere más datos
- **Caso de uso**: Datasets muy grandes

---

## 🔨 Ejemplo de Código: Gradient Boosting

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# 1. Cargar datos
df = pd.read_csv("Glucosa_Unique_Completo.csv")

# 2. Crear categorías
def clasificar_glucosa(valor):
    if valor <= 100:
        return "Normal"
    elif valor <= 125:
        return "Prediabetes"
    else:
        return "Diabetes"

df["Categoria_Glucosa"] = df["Resultado"].apply(clasificar_glucosa)

# 3. Seleccionar features
features_seleccionadas = [
    "Edad_Años", "peso", "talla", "imc", "tas", "tad", "Categoria_Glucosa"
]
target = "Resultado"

df_limpio = df.dropna(subset=features_seleccionadas + [target]).copy()
X = df_limpio[features_seleccionadas]
y = df_limpio[target]

# 4. Preprocesador
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 5. Pipeline con Gradient Boosting
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# 6. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 7. Entrenar
model.fit(X_train, y_train)

# 8. Evaluar
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
rmse_train = (mean_squared_error(y_train, y_pred_train))**0.5
mae_train = mean_absolute_error(y_train, y_pred_train)

r2_test = r2_score(y_test, y_pred_test)
rmse_test = (mean_squared_error(y_test, y_pred_test))**0.5
mae_test = mean_absolute_error(y_test, y_pred_test)

print("="*50)
print("EVALUACIÓN DEL MODELO GRADIENT BOOSTING")
print("="*50)
print(f"[ENTRENAMIENTO] R²={r2_train:.3f} | RMSE={rmse_train:.2f} mg/dL | MAE={mae_train:.2f} mg/dL")
print(f"[PRUEBA       ] R²={r2_test:.3f} | RMSE={rmse_test:.2f} mg/dL | MAE={mae_test:.2f} mg/dL")
print("="*50)

# 9. Guardar
joblib.dump(model, "modelo_gradient_boosting.joblib")
print(f"✅ Modelo guardado")
```

---

## 💾 Cómo Usar un Modelo Entrenado

```python
import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load("modelo_gradient_boosting.joblib")

# Preparar datos nuevos
X_nuevo = pd.DataFrame({
    "Edad_Años": [45],
    "peso": [75],
    "talla": [170],
    "imc": [26],
    "tas": [120],
    "tad": [80],
    "Categoria_Glucosa": ["Normal"]
})

# Predecir
prediccion = modelo.predict(X_nuevo)
print(f"Glucosa predicha: {prediccion[0]:.2f} mg/dL")
```

---

## 📊 Interpretación de Métricas

### R² (Coeficiente de Determinación)
- **Rango**: 0 a 1 (más alto es mejor)
- **Interpretación**: Proporción de varianza explicada
- **Ejemplo**: R²=0.86 = El modelo explica el 86% de la variabilidad

### RMSE (Root Mean Squared Error)
- **Unidad**: mg/dL
- **Interpretación**: Error promedio esperado
- **Ventaja**: Penaliza errores grandes

### MAE (Mean Absolute Error)
- **Unidad**: mg/dL
- **Interpretación**: Error absoluto promedio
- **Ventaja**: Más intuitivo que RMSE

---

## 📁 Estructura de Archivos

```
proyecto-glucosa-ml/
│
├── README.md                                    # Este archivo
├── Glucosa_Unique_Completo.csv                  # Dataset
│
├── modelos/
│   ├── train_linear_regression.py
│   ├── train_ridge_regression.py
│   ├── train_lasso_regression.py
│   ├── train_random_forest.py
│   ├── train_gradient_boosting.py               # Ejemplo del código
│   ├── train_svm.py
│   └── train_mlp.py
│
└── modelos_entrenados/
    ├── modelo_linear_regression.joblib
    ├── modelo_ridge_regression.joblib
    ├── modelo_lasso_regression.joblib
    ├── modelo_random_forest.joblib
    ├── modelo_gradient_boosting.joblib
    ├── modelo_svm.joblib
    └── modelo_mlp.joblib
```

---

## ✅ Ventajas de Esta Estructura

- ✅ **Modular**: Fácil agregar nuevos modelos
- ✅ **Reproducible**: random_state garantiza resultados idénticos
- ✅ **Escalable**: Preprocesamiento automatizado
- ✅ **Reutilizable**: Pipeline encapsulado
- ✅ **Comparable**: Todos los modelos con misma metodología

---

## ⚠️ Limitaciones

- ⚠️ Requiere datos de calidad bien estructurados
- ⚠️ Error de ±12-15 mg/dL requiere confirmación médica
- ⚠️ No sustituye diagnóstico profesional
- ⚠️ Modelos específicos para predicción de glucosa

---

## 🚀 Mejoras Futuras

1. Validación cruzada K-Fold
2. Hiperparámetro tuning automático (Grid Search)
3. Feature importance analysis
4. Comparativa visual con gráficos
5. API REST para despliegue
6. Dashboard interactivo (Streamlit)

---

## 📖 Referencias Clínicas

- OMS (2006): Definición y diagnóstico de diabetes mellitus
- ADA Standards of Care: Criterios de clasificación
- Friedman (2001): Gradient Boosting Machines
- Chen & Guestrin (2016): XGBoost - Scalable Tree Boosting

---

## 📧 Contacto

Para preguntas o sugerencias: [tu email]

---

**Última actualización**: Octubre 2025
**Versión**: 1.0
