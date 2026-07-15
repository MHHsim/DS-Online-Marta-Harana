# Proyecto_Modelo_machine_learning

# Predicción de aceptación de depósito bancario (Bank Marketing)

## 1. Descripción del proyecto
El objetivo del proyecto es construir un modelo supervisado capaz de predecir si un cliente aceptará un depósito bancario **antes de realizar la llamada**.  
Esto permite optimizar campañas de marketing, priorizar clientes con mayor probabilidad de conversión y reducir costes operativos.

El problema se formula como una **clasificación binaria** con fuerte **desbalanceo** (clase positiva minoritaria).

---

## 2. Dataset
- Fuente: *Bank Marketing Dataset (UCI Machine Learning Repository)*  
- Nº de observaciones: ~45.000  
- Nº de variables: 16  
- Target: `y` (yes/no → aceptación del depósito)

### Principales retos del dataset:
- **Desbalanceo severo** → métricas estándar como accuracy no son adecuadas.  
- **Valores centinela** (`pdays = -1`) que requieren reinterpretación semántica.  
- **Categorías “unknown”** con significados distintos según la variable.  
- **Variable `duration`** que introduce **data leakage** (solo se conoce tras la llamada).

---

## 3. Feature Engineering

### Eliminación de variables con fuga de información
- `duration` se elimina completamente.  
  Su valor depende del resultado de la llamada → **no puede usarse para predecir antes de llamar**.

### Reinterpretación de valores centinela
- `pdays = -1` → indica *ausencia de contacto previo*.  
- Se transforma en variable binaria:  
  **0 = sin contacto previo**, **1 = contacto previo**.

### Alineación semántica entre variables
- `poutcome = unknown` → se recodifica como `no_previous_contact`.  
- Coincide conceptualmente con `pdays = -1`.

### Imputación de valores faltantes reales
- `job = unknown` aparece en solo 0.6%.  
- Se imputa con la **moda del conjunto de entrenamiento** para evitar distorsión.

### Codificación de variables binarias
- `default`, `housing`, `loan` → se convierten a **0/1**.

### Conservación de categorías válidas
- `education` y `contact` mantienen `unknown` como categoría legítima.

---

## 4. Preparación del modelado

### Separación Train/Test
Se realiza un split estratificado para preservar la proporción de clases.

### Identificación de tipos de variables
- Variables numéricas  
- Variables categóricas  
- Variables binarias ya transformadas

---

## 5. Pipelines para modelos

Se construyen **3 pipelines independientes**, optimizados para cada familia de modelos:

### Modelos que requieren escalado + OneHot
- Logistic Regression  
- KNN  
- SVM  

Incluyen:
- `StandardScaler` para numéricas  
- `OneHotEncoder` para categóricas

### Modelos que no requieren escalado pero sí OneHot
- DecisionTree  
- RandomForest  
- XGBoost  
- LightGBM  

Incluyen:
- Numéricas en *passthrough*  
- `OneHotEncoder` para categóricas

### CatBoost
- No requiere escalado  
- No requiere OneHot  
- Maneja categóricas de forma nativa  
- Internamente gestiona el desbalanceo

---

## 6. Modelos base con tratamiento del desbalanceo

Dado el desbalanceo del dataset, los modelos base se entrenan con:

- `class_weight="balanced"` para modelos lineales y árboles clásicos  
- `scale_pos_weight = (negativos / positivos)` para XGBoost  
- `class_weight="balanced"` para LightGBM  
- CatBoost maneja el desbalanceo internamente

Esto garantiza que el modelo base ya sea competitivo y adecuado para el problema.

---

## 7. Evaluación con Cross‑Validation

Se utiliza **Cross‑Validation estratificada (cv=5)** sobre el conjunto de entrenamiento.

### Métrica principal:
- **F1-score**  
  Adecuada para datasets desbalanceados y campañas donde interesa maximizar la detección de la clase positiva sin perder precisión.

### Métricas secundarias:
- ROC-AUC  
- PR-AUC (más informativa en desbalanceo extremo)

El mejor modelo base se selecciona según el **F1 promedio en CV**.

---

## 8. Optimización de hiperparámetros

Solo se optimiza el **modelo con mejor rendimiento base**.  
Se utiliza `GridSearchCV` o `RandomizedSearchCV` según el coste computacional.

La optimización se realiza **exclusivamente sobre el conjunto de entrenamiento**, manteniendo el test aislado para evaluación final.

---

## 9. Evaluación final en test

El modelo optimizado se evalúa sobre el conjunto de test mediante:

- **F1-score**  
- **ROC-AUC**  
- **PR-AUC**  
- **Matriz de confusión**  
- **Curva ROC**  
- **Curva Precision-Recall**

Esta evaluación representa el rendimiento **real** del modelo en un escenario de producción.

---

## 10. Conclusiones técnicas

- El tratamiento del desbalanceo es crítico para obtener modelos útiles.  
- La eliminación de `duration` evita leakage y mejora la validez del modelo.  
- La ingeniería de variables (`pdays`, `poutcome`) aporta señal relevante.  
- Los modelos basados en árboles y boosting suelen ser más robustos en datasets mixtos.  
- CatBoost destaca por su manejo nativo de categóricas y estabilidad en CV.  
- El flujo de trabajo garantiza reproducibilidad, ausencia de fuga de información y evaluación justa.

---

## 11. Estructura del repositorio

El proyecto debe entregarse en un repositorio público con el nombre: `ML_[temática-del-proyecto]`

```
├── src/
│   ├── data_sample/    # Archivos de datos de muestra (máx. 100MB)
│   ├── img/            # Imágenes utilizadas en el proyecto
│   ├── models/         # Modelos guardados en formato pickle o joblib
│   ├── notebooks/      # Notebooks de desarrollo y pruebas
│   ├── utils/          # Módulos y funciones auxiliares
├── main.ipynb          # Notebook final del pipeline de ML
├── Presentacion.pdf    # Documento soporte de la exposición
├── README.md           # Fichero README resumen del proyecto
├── requirements.txt    # Fichero con las dependencias usadas en el proyecto y reproducible

---


## 12. Requisitos

- pandas  
- numpy  
- scikit-learn  
- xgboost  
- lightgbm  
- catboost  
- matplotlib / seaborn  

---

## 13. Ejecución del proyecto

1. Instalar dependencias  
2. Ejecutar el notebook principal o script en `src/`  
3. Reproducir el flujo completo:  
   - Feature engineering  
   - Construcción de pipelines  
   - Entrenamiento de modelos base  
   - Cross‑Validation  
   - Optimización  
   - Evaluación final






# 📘 README.md

## 🧩 Descripción del problema
Explica brevemente:
- Contexto del negocio  
- Necesidad detectada  
- Objetivo del proyecto  

## 📊 Dataset utilizado
- Descripción breve del dataset  
- Indicar si es público o privado  
- Cómo acceder a él (enlace, instrucciones, credenciales si aplica)  
- Variables principales y tamaño aproximado  

## 🤖 Solución adoptada
- Tipo de problema (clasificación, regresión, clustering…)  
- Modelos probados  
- Modelo final seleccionado y justificación  
- Descripción del pipeline y etapas principales  

## 📁 Estructura del repositorio

    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    ├── src/
    │   ├── preprocessing/
    │   ├── models/
    │   └── utils/
    ├── reports/
    │   └── figures/
    ├── README.md
    └── requirements.txt

## 🛠️ Tecnologías utilizadas
- Python  
- pandas, numpy, scikit-learn  
- XGBoost, LightGBM, CatBoost  
- Matplotlib, Seaborn  
- Jupyter Notebook  
- Git / GitHub  
- Otros (Docker, FastAPI, etc.)  

## ▶️ Instrucciones de reproducción
1. Clonar el repositorio  
2. Crear entorno virtual  
3. Instalar dependencias: `pip install -r requirements.txt`  
4. Ejecutar notebooks o scripts  
5. Configurar rutas necesarias o variables de entorno  

## 📈 Principales resultados
- Métricas clave del modelo  
- Comparación entre modelos probados  
- Conclusiones principales  
- Limitaciones del proyecto  
- Posibles mejoras futuras  

## Autores
Incluye nombres y enlaces:
- **Nombre del autor 1** — GitHub: [https://github.com/usuario1](https://github.com/usuario1) — LinkedIn: [https://linkedin.com/in/usuario1](https://linkedin.com/in/usuario1)  
- **Nombre del autor 2** — GitHub: [https://github.com/usuario2](https://github.com/usuario2) — LinkedIn: [https://linkedin.com/in/usuario2](https://linkedin.com/in/usuario2)  

## 🔒 Datos sensibles
- Indica si el proyecto contiene datos personales o confidenciales.  
- Explica las medidas tomadas para protegerlos (anonimización, eliminación de campos, uso de datos sintéticos…).  
- Señala si se requiere aprobación antes de publicar el repositorio.  
- Aclara que no se incluye ningún archivo que contenga información privada.  

## 🎥 Nota importante
El archivo de exposición (vídeo MP4) no debe incluirse en el repositorio.  
Debe entregarse únicamente a través del Campus Virtual.
