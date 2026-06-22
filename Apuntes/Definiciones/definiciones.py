import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway



# import sys
# sys.path.append(r"c:\Users\Usuario\Documents\GitHub\DS-Online-Marta-Harana\Apuntes")
# from definiciones import...

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadístico descriptivo de un DataFrame.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        pd.DataFrame: DataFrame con una fila por columna del input y las
        siguientes columnas: 'tipo', 'porcentaje_nulos', 'valores_unicos',
        'porcentaje_cardinalidad'.
        Retorna None si el input no es un DataFrame válido.
    """
    
    # Comprobación de que si es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el objeto proporcionado no es un DataFrame.")
        return None

    # Crear el DataFrame resultado
    resultado = pd.DataFrame(index=df.columns)

    # Tipo de dato
    resultado["tipo"] = df.dtypes.astype(str)

    # Porcentaje de nulos
    resultado["porcentaje_nulos"] = (df.isna().mean() * 100).round(2)

    # Valores únicos
    resultado["valores_unicos"] = df.nunique()

    # Porcentaje de cardinalidad
    resultado["porcentaje_cardinalidad"] = ((df.nunique() / len(df)) * 100).round(2)


    return resultado

def tipifica_variables(df: pd.DataFrame, umbral_categorica: int, umbral_continua: float) -> pd.DataFrame:
    """
    Clasifica las variables de un DataFrame según su cardinalidad y porcentaje de cardinalidad.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        umbral_categorica: Umbral para categorizar las variables categóricas
        umbral_continua: Float

    Retorna:
        pd.DataFrame: DataFrame con dos columnas: 'nombre_variable', 'tipo_sugerido'
        Retorna None si el input no es un DataFrame válido.
        Retorna None si el umbral_categorica no es un tipo válido (int).
        Retorna None si el umbral_continua no es un tipo válido (float)
    """
    
    # Comprobación de DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el objeto proporcionado no es un DataFrame.")
        return None
    
    # Comprobación umbral_categorica, debe ser entero positivo
    if not isinstance(umbral_categorica, int) or umbral_categorica <= 0:
        print("Error: umbral_categorica debe ser un entero positivo.")
        return None

    # comprobación umbral_continua, debe ser float entre 0 y 100
    if not isinstance(umbral_continua, float) or not (0 <= umbral_continua <= 100):
        print("Error: umbral_continua debe ser un float entre 0 y 100.")
        return None
    
    
    # Cardinalidad y porcentaje_cardinalidad
    cardinalidad = df.nunique()
    porcentaje_cardinalidad = df.nunique() / len(df) * 100

    # Clasificación variables
    tipos = []

    for col in df.columns:
        card = cardinalidad[col]
        pct = porcentaje_cardinalidad[col]

        if card == 2:
            tipo = "Binaria"

        elif card < umbral_categorica:
            tipo = "Categórica"

        elif card >= umbral_categorica and pct >= umbral_continua:
            tipo = "Numérica Continua"

        else:
            tipo = "Numérica Discreta"

        tipos.append(tipo)
    
    # DF resultado
    resultado = pd.DataFrame({
        "nombre_variable": df.columns,
        "tipo_sugerido": tipos
    })

    return resultado


from sklearn.feature_selection import f_classif
import pandas as pd

def calcular_anova(df, target):
    """
    Calcula ANOVA entre todas las variables numéricas del dataframe y un target categórico.
    
    Parámetros:
        df (DataFrame): tu dataset
        target (str): nombre de la columna target categórica
    
    Retorna:
        DataFrame ordenado por p_value ascendente
    """
    
    # 1. Seleccionar solo columnas numéricas
    features_num = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # 2. Separar X e y
    X = df[features_num]
    y = df[target]
    
    # 3. Calcular ANOVA
    F_scores, p_values = f_classif(X, y)
    
    # 4. Crear tabla de resultados
    anova_results = pd.DataFrame({
        "feature": features_num,
        "F_score": F_scores,
        "p_value": p_values
    }).sort_values("p_value")
    
    return anova_results

from scipy import stats

def get_features_num_classification(df, target, pvalue_rango=0.05):
    """
    Selección de features numéricas relevantes para un modelo de 
    clasificación usando el método estadístico ANOVA.

    Parámetros:
        - df : DataFrame
        - target : str - Nombre de la columna objetivo.
        - pvalue_rango : float - Umbral de significancia (default=0.05).

    Retorna:
    --------
    features_num_sel : Lista de nombres de features numéricas seleccionadas.
    """
    # Separar X e y
    y = df[target]
    X = df.drop(columns=[target])

    # Seleccionar solo columnas numéricas
    features_num = X.select_dtypes(include=["int64", "float64"])

    # Si no hay columnas numéricas retorna una lista vacía
    if features_num.shape[1] == 0:
        return []

    # Creamos una lista vacía para que se guarden las features seleccionadas
    features_num_sel = []

    # Para cada columna numérica aplicamos el método y calculamos p_value
    for col in features_num.columns:

        # Crear lista de grupos: una lista por cada clase del target
        grupos = [df[df[target] == clase][col].dropna() 
                  for clase in df[target].unique()]

        # Método ANOVA
        f_val, p_val = stats.f_oneway(*grupos)

        # Seleccionar si p < umbral, se rechaza la hipótesis nula (la variable es dependiente)
        if p_val < pvalue_rango:
            features_num_sel.append(col)

    return features_num_sel

from scipy.stats import chi2_contingency


def get_features_cat_classification(df, target, pvalue_rango=0.05):
    """
    Selección de features categórcias relevantes para un modelo de 
    clasificación usando el método estadístico Chi-cuadrado.

    Parámetros:
        - df : DataFrame
        - target : str - Nombre de la columna objetivo.
        - pvalue_rango : float - Umbral de significancia (default=0.05).

    Retorna:
    --------
    features_cat_sel : Lista de nombres de features numéricas seleccionadas.
    """
    # Separar X e y
    y = df[target]
    X = df.drop(columns=[target])
    # Convertir columnas string (StringDtype) a object para evitar errores con SciPy
    X = X.astype({col: "object" for col in X.select_dtypes(include="string").columns})

    # Seleccionar solo columnas categóricas
    features_cat = X.select_dtypes(include=["object", "category"])

    # Si no hay columnas numéricas retorna una lista vacía
    if features_cat.shape[1] == 0:
        return []

    # Creamos una lista vacía para que se guarden las features seleccionadas
    features_cat_sel = []

    # Para cada columna cat aplicamos el método y calculamos p_value
    for col in features_cat.columns:

        # Crear tabla de contingencia
        tabla_contingencia = pd.crosstab(df[col], df[target])

        # Test Chi-cuadrado
        chi2, p_val, dof, expected = chi2_contingency(tabla_contingencia)

        # Seleccionar si p < umbral, se rechaza la hipótesis nula (la variable es dependiente)
        if p_val < pvalue_rango:
            features_cat_sel.append(col)

    return features_cat_sel


from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np

def get_corr_features(
    df: pd.DataFrame,
    features_num: list,
    target: str,
    umbral_corr: float = 0.2,
    pvalue: float = None
):
    """
    Calcula correlaciones Pearson y Spearman entre las features numéricas y el target.
    Devuelve:
        - tabla con ambas correlaciones
        - lista de features seleccionadas según el umbral
    """

    resultados = []

    for col in features_num:
        datos = df[[col, target]].dropna()

        # evitar columnas constantes
        if datos[col].nunique() < 2:
            continue

        # Pearson
        r_p, p_p = pearsonr(datos[col], datos[target])

        # Spearman
        r_s, p_s = spearmanr(datos[col], datos[target])

        resultados.append([col, r_p, p_p, r_s, p_s])

    tabla = pd.DataFrame(
        resultados,
        columns=['feature', 'pearson', 'p_pearson', 'spearman', 'p_spearman']
    ).set_index('feature')

    # Selección de features según umbral y pvalue
    if pvalue is None:
        seleccionadas = tabla[
            (tabla['pearson'].abs() > umbral_corr) |
            (tabla['spearman'].abs() > umbral_corr)
        ].index.tolist()
    else:
        seleccionadas = tabla[
            (
                (tabla['pearson'].abs() > umbral_corr) &
                (tabla['p_pearson'] < pvalue)
            ) |
            (
                (tabla['spearman'].abs() > umbral_corr) &
                (tabla['p_spearman'] < pvalue)
            )
        ].index.tolist()

    return tabla, seleccionadas
