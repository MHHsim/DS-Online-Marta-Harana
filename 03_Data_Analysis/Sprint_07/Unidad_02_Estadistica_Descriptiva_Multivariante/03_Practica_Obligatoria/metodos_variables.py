# metodo chi_cuadrado (categorica, categorica)
tabla_contingencia = pd.crosstab(df['Admission'], df['Major'])

tabla_contingencia


from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(tabla_contingencia)


print("Valor Chi-Cuadrado:", chi2)
print("P-Value:", p)
print("Grados de Libertad:", dof)
print("Tabla de Frecuencias Esperadas:\n", expected)



# Método de la prueba U de Mann-Whitney: (categórica binaria, numérica)
variable_categorica = "alive"
valor_1_categorica = "yes"
valor_2_categorica = "no"
variable_numerica = "fare"
df_resultado = df
from scipy.stats import mannwhitneyu 

# Hipótesis nula, no existe relación entre las dos variables

#Separamos los datos en dos grupos según la variable categórica
grupo_a = df_resultado.loc[df_resultado[variable_categorica] == valor_1_categorica][variable_numerica]
grupo_b = df_resultado.loc[df_resultado[variable_categorica] == valor_2_categorica][variable_numerica]

#Aplicamos la prueba y mostramos los resultados
u_stat, p_valor = mannwhitneyu(grupo_a, grupo_b)

print("Estadístico U:", u_stat)
print("Valor p:", p_valor)

#Si el resultado de p es mayor a 0.05 -- No hay evidencia estadística de rechazar la hipótesis nula
#Si el resultado de p es menor a 0.05 -- Hipóteis alternativa - Hay relación entre las dos variables que estamos analizando



# Método anova (variable categórica no binaria - variable numérica)
from scipy import stats

grupos = df_tips['size'].unique()  # Obtener los valores únicos de la columna categórica
var_cat_grupo = [df_tips[df_tips['size'] == grupo]['tip'] for grupo in grupos] # Obtenemos la variable numérica por valor de la categórica y los incluimos en una lista
var_cat_grupo


f_val, p_val = stats.f_oneway(*var_cat_grupo) # El método * (igual que cuando vimos *args hace mil años) 
                                                    # lo que hace es separar todos los elementos de la lista y pasarselos como argumento a la función
print("Valor F:", f_val)
print("Valor p:", p_val)




# Metodo ANOVA (variable categórica NO binaria vs numérica)
variable_categorica = "size"
variable_numerica = "tip"
df_resultado = df_tips

from scipy import stats

# 1. Obtener los valores únicos de la variable categórica
grupos = df_resultado[variable_categorica].unique()

# 2. Crear una lista donde cada elemento es la serie numérica filtrada por cada categoría
listas_grupos = [
    df_resultado[df_resultado[variable_categorica] == grupo][variable_numerica]
    for grupo in grupos
]

# 3. Aplicar ANOVA
f_val, p_val = stats.f_oneway(*listas_grupos)

# 4. Mostrar resultados
print("Valor F:", f_val)
print("Valor p:", p_val)

#P < 0.05  →  Rechazamos la hipótesis nula (H₀).  
#            Hay evidencia estadística de relación o diferencia entre las variables.
 
#P > 0.05  →  No rechazamos la hipótesis nula (H₀).  
#             No hay evidencia estadística suficiente para afirmar relación o diferencia.