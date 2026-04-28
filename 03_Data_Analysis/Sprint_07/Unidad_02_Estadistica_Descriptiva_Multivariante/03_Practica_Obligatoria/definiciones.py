# cardinalidad para clasificar el tipo de variables
def cardinalidad(df_in, umbral_categoria, umbral_continua):
    
    resultados = []

    for col in df_in.columns:
        # Cardinalidad absoluta
        card = df_in[col].nunique(dropna=True)
        
        # Porcentaje de cardinalidad
        pct = card / len(df_in) * 100
        
        # Clasificación según reglas
        if card == 2:
            tipo = "Binaria"
        
        elif card < umbral_categoria:
            tipo = "Categórica"
        
        else:
            # card >= umbral_categoria → mirar porcentaje
            if pct >= umbral_continua:
                tipo = "Numerica Continua"
            else:
                tipo = "Numerica Discreta"
        
        resultados.append({
            "columna": col,
            "cardinalidad": card,
            "pct_cardinalidad": pct,
            "clasificacion": tipo
        })
    
    return pd.DataFrame(resultados)




# para variables numéricas
def analisis_descriptivo(df, col_total="total"):
    """
    Genera un análisis descriptivo completo para todas las columnas numéricas de un DataFrame.
    
    Incluye:
    - Medidas de tendencia central
    - Cuartiles
    - IQR
    - Mínimo y máximo
    - Rango total
    - Coeficiente de variación (CV)
    - Conversión opcional a valores absolutos multiplicando por 'col_total'
    """

    # 1. Selección de columnas numéricas
    num_cols = df.select_dtypes(include="number").columns

    # 2. Copia del df y conversión a valores absolutos (si aplica)
    df_abs = df.copy()
    if col_total in num_cols:
        for col in num_cols:
            if col != col_total:
                df_abs[col] = df_abs[col] * df_abs[col_total]

    # 3. Tabla de medidas descriptivas
    tabla = pd.DataFrame({
        "media": df_abs[num_cols].mean(),
        "mediana": df_abs[num_cols].median(),
        "Q1": df_abs[num_cols].quantile(0.25),
        "Q2": df_abs[num_cols].quantile(0.50),
        "Q3": df_abs[num_cols].quantile(0.75),
        "IQR": df_abs[num_cols].quantile(0.75) - df_abs[num_cols].quantile(0.25),
        "min": df_abs[num_cols].min(),
        "max": df_abs[num_cols].max(),
        "rango_total": df_abs[num_cols].max() - df_abs[num_cols].min()
    })

    # 4. Función interna para CV
    def coef_variacion(serie):
        media = serie.mean()
        std = serie.std()
        return std / media

    # 5. Añadir CV como columna adicional
    tabla["cv"] = df_abs[num_cols].apply(coef_variacion)

    return tabla
