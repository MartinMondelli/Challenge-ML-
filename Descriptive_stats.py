import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

## Import data as a panda dataframe
df_train_in = pd.read_csv("train_features_local.csv", index_col=0)
y_train = pd.read_csv("y_train_local.csv", index_col=0)
df_test = pd.read_csv("df_test_local.csv", index_col=0)
# Features polinómicas y log
df_train_in['length_2'] = df_train_in['length'] ** 2
df_train_in['length_3'] = df_train_in['length'] ** 3
df_train_in['length_ln'] = np.log(df_train_in['length'].replace(0, np.nan)).replace([-np.inf, np.inf], np.nan).fillna(-1)
df_train_in['budget_2'] = df_train_in['budget'] ** 2
df_train_in['budget_3'] = df_train_in['budget'] ** 3
df_train_in['budget_ln'] = np.log(df_train_in['budget'].replace(0, np.nan)).replace([-np.inf, np.inf], np.nan).fillna(-1)
df_train_in['popularity_score_2'] = df_train_in['popularity_score'] ** 2
df_train_in['popularity_score_3'] = df_train_in['popularity_score'] ** 3
df_train_in['popularity_score_ln'] = np.log(df_train_in['popularity_score'].replace(0, np.nan)).replace([-np.inf, np.inf], np.nan).fillna(-1)

"""
#======================================================

# Scatter simple
#Budget - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_in["budget"], y=y_train["revenue"])
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.title("Scatter plot: Budget vs Revenue")
plt.show()
#Parece que hay correlacion positiva, confirmado por Spearman y Pearson
#(hay que ver con log)

#length - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=np.log(df_train_in["length"]), y=y_train["revenue"])
plt.xlabel("length")
plt.ylabel("Revenue")
plt.title("Scatter plot: length vs Revenue")
plt.show()
#Parece que no hay correlacion, muy baja ya sea con Pearson o Spearman

#popularity_score - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_in["popularity_score"], y=y_train["revenue"])
plt.xlabel("popularity_score")
plt.ylabel("Revenue")
plt.title("Scatter plot: popularity_score vs Revenue")
plt.show()
#Hay correlacion pero poco fuerte, hay que ver con log

# Scatter ^2
#Budget - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_in["budget_2"], y=y_train["revenue"])
plt.xlabel("Budget^2")
plt.ylabel("Revenue")
plt.title("Scatter plot: Budget vs Revenue")
plt.show()
#Parece que hay correlacion positiva, confirmado por Spearman y Pearson
#(hay que ver con log)

#length - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=np.log(df_train_in["length_2"]), y=y_train["revenue"])
plt.xlabel("length^2")
plt.ylabel("Revenue")
plt.title("Scatter plot: length vs Revenue")
plt.show()
#Parece que no hay correlacion, muy baja ya sea con Pearson o Spearman

#popularity_score - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_in["popularity_score_2"], y=y_train["revenue"])
plt.xlabel("popularity_score^2")
plt.ylabel("Revenue")
plt.title("Scatter plot: popularity_score vs Revenue")
plt.show()

# Scatter ^3
#Budget - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_in["budget_3"], y=y_train["revenue"])
plt.xlabel("Budget^3")
plt.ylabel("Revenue")
plt.title("Scatter plot: Budget vs Revenue")
plt.show()
#Parece que hay correlacion positiva, confirmado por Spearman y Pearson
#(hay que ver con log)

#length - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=np.log(df_train_in["length_3"]), y=y_train["revenue"])
plt.xlabel("length^3")
plt.ylabel("Revenue")
plt.title("Scatter plot: length vs Revenue")
plt.show()
#Parece que no hay correlacion, muy baja ya sea con Pearson o Spearman

#popularity_score - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_in["popularity_score_3"], y=y_train["revenue"])
plt.xlabel("popularity_score^3")
plt.ylabel("Revenue")
plt.title("Scatter plot: popularity_score vs Revenue")
plt.show()

# Scatter ^2
#Budget - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_in["budget_ln"], y=y_train["revenue"])
plt.xlabel("Budget_ln")
plt.ylabel("Revenue")
plt.title("Scatter plot: Budget vs Revenue")
plt.show()
#Parece que hay correlacion positiva, confirmado por Spearman y Pearson
#(hay que ver con log)

#length - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=np.log(df_train_in["length_ln"]), y=y_train["revenue"])
plt.xlabel("length_ln")
plt.ylabel("Revenue")
plt.title("Scatter plot: length vs Revenue")
plt.show()
#Parece que no hay correlacion, muy baja ya sea con Pearson o Spearman

#popularity_score - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_in["popularity_score_ln"], y=y_train["revenue"])
plt.xlabel("popularity_score_ln")
plt.ylabel("Revenue")
plt.title("Scatter plot: popularity_score vs Revenue")
plt.show()

#Ultima prueba para length con escala

# Columnas a centrar y reducir
cols_to_scale = ['length',
                 'budget',
                 'popularity_score']

df_train_scaled = df_train_in.copy()

for col in cols_to_scale:
    mean = df_train_scaled[col].mean()        # media
    std = df_train_scaled[col].std()          # desviación estándar
    df_train_scaled[col + '_scaled'] = (df_train_scaled[col] - mean) / std

# Scatter Escalado
#Budget - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_scaled["budget_scaled"], y=y_train["revenue"])
plt.xlabel("Budget_scaled")
plt.ylabel("Revenue")
plt.title("Scatter plot: Budget vs Revenue")
plt.show()
#Parece que hay correlacion positiva, confirmado por Spearman y Pearson
#(hay que ver con log)

#length - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=np.log(df_train_scaled["length_scaled"]), y=y_train["revenue"])
plt.xlabel("length_scaled")
plt.ylabel("Revenue")
plt.title("Scatter plot: length vs Revenue")
plt.show()
#Parece que no hay correlacion, muy baja ya sea con Pearson o Spearman

#popularity_score - revenue
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_scaled["popularity_score_scaled"], y=y_train["revenue"])
plt.xlabel("popularity_score_scaled")
plt.ylabel("Revenue")
plt.title("Scatter plot: popularity_score vs Revenue")
plt.show()

#======================================================

#Matriz dispersion
features = ["budget", "popularity_score", "length", "revenue"]
df_plot = df_train_in.copy()
df_plot["revenue"] = y_train.values.ravel()

sns.pairplot(df_plot[features], diag_kind="kde")
plt.show()

#Calcular Corr
df_plot = df_plot[features]
# Pearson
pearson_corr = df_plot.corr(method="pearson")
print("Pearson correlation:\n", pearson_corr)

# Spearman
spearman_corr = df_plot.corr(method="spearman")
print("Spearman correlation:\n", spearman_corr)

#Grafico corr
plt.figure(figsize=(10,8))
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson Correlation Heatmap")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman Correlation Heatmap")
plt.show()
"""

##======================================================
#NAs
"""
print("Number NAs")
print("Budget: ",df_train_in['budget'].isna().sum())
print("Length: ",df_train_in['length'].isna().sum())
print("Location NA length: \n", df_train_in.loc[df_train_in['length'].isna()])
#Observation n# 481 Королёв in Russian
print("Popularity score: ",df_train_in['popularity_score'].isna().sum())
print("Date: ",df_train_in['date'].isna().sum())
print("Cast", df_train_in['cast'].isna().sum())
print("Collection: ", df_train_in['collection'].isna().sum())
#We can fairly suppose that if collection is NA then probably no collection (else drop it)
print("Company: ",df_train_in['company'].isna().sum())
print("Location NA comapny: \n", df_train_in.loc[df_train_in['company'].isna()])
#They are really NA War, Inc is from Universal for example but figures as NA
#Same with "The Good Doctor", it is from Magnolia Pictures
#Drop it? Majority class?
print("Webpage: ", df_train_in['webpage'].isna().sum())
#We can fairly suppose that if webpage is NA then probably no collection (else drop it)
print("Language: ",df_train_in['language'].isna().sum())
print("Country: ", df_train_in['country'].isna().sum())
#We didn't use this variable for yet
"""
##======================================================
"""
#Outliers
sns.boxplot(x=df_train_in['budget'])
plt.show()
sns.boxplot(x=df_train_in['length'])
plt.show()
sns.boxplot(x=df_train_in['popularity_score'])
plt.show()

#Identificarlos y contarlos
#Budget
Q1 = df_train_in['budget'].quantile(0.25)
Q3 = df_train_in['budget'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_budget = df_train_in[(df_train_in['budget'] < lower_bound) | (df_train_in['budget'] > upper_bound)]
print("Outliers budget: ",outliers_budget)
#Hay 148 (7.4% de los 2000, >5% problema)

#Length
Q1 = df_train_in['length'].quantile(0.25)
Q3 = df_train_in['length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_length = df_train_in[(df_train_in['length'] < lower_bound) | (df_train_in['length'] > upper_bound)]
print("Outliers length: ",outliers_length)
#Hay 83 (4.2% de los 2000 <5% entonces OK)$

#Popularity score
Q1 = df_train_in['popularity_score'].quantile(0.25)
Q3 = df_train_in['popularity_score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_popularity = df_train_in[(df_train_in['popularity_score'] < lower_bound) | (df_train_in['popularity_score'] > upper_bound)]
print("Outliers popularity score: ",outliers_popularity)
#Hay 43 (2.1% de los 2000 <5% entonces OK)

#Hacemos windsorization (sino podemos reemplazar por mean)
"""
from scipy.stats.mstats import winsorize

df_train_in['budget_wind'] = np.array(winsorize(df_train_in['budget'], limits=[0.05, 0.05]))
df_train_in['length_wind'] = np.array(winsorize(df_train_in['length'], limits=[0.02, 0.02]))
df_train_in['popularity_score_wind'] = np.array(winsorize(df_train_in['popularity_score'], limits=[0.01, 0.01]))#Aplicamos margenes para llegar a 2% del total

#Chequeamos boxplot
sns.boxplot(x=df_train_in['budget_wind'])
plt.show()
sns.boxplot(x=df_train_in['length_wind'])
plt.show()
sns.boxplot(x=df_train_in['popularity_score_wind'])
plt.show()

#Chequeamos numero total
#Budget
Q1_w = np.percentile(df_train_in['budget_wind'], 25)
Q3_w = np.percentile(df_train_in['budget_wind'], 75)
IQR_w = Q3_w - Q1_w
lower_bound_w = Q1_w - 1.5*IQR_w
upper_bound_w = Q3_w + 1.5*IQR_w

outliers_budget_wind = df_train_in[(df_train_in['budget_wind'] < lower_bound_w) | (df_train_in['budget_wind'] > upper_bound_w)]
print("Outliers budget (Winsorized):", len(outliers_budget_wind))

#Length
Q1 = df_train_in['length_wind'].quantile(0.25)
Q3 = df_train_in['length_wind'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_length_wind = df_train_in[(df_train_in['length_wind'] < lower_bound) | (df_train_in['length_wind'] > upper_bound)]
print("Outliers length (Wind): ",outliers_length_wind)

#Popularity score
Q1 = df_train_in['popularity_score_wind'].quantile(0.25)
Q3 = df_train_in['popularity_score_wind'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_popularity_wind = df_train_in[(df_train_in['popularity_score_wind'] < lower_bound) | (df_train_in['popularity_score_wind'] > upper_bound)]
print("Outliers popularity score (Wind): ",outliers_popularity_wind)

##======================================================
## Notas
##======================================================

#Etapas a hacer en Programa:

#1 - A - Utilizar para budget su valor original (o centrado y reducido)
#1 - B - Ver si no sacamos length (o ponerlo en x^3
#1 - C - Poner popularity_score en ln

#2 - A - Para company dejar la dummy que esta pero agregar una dummy para NA en company
#2 - B - Webpage como proxy de marketing (?) muchos NAs

#3 - A - Hay muchos outliers, sobre todo para budget, conviene hacer winsorisation (copiar y pegar codigo)

#4 - A - Conviene centrar y reducir variables antes (?) / cambiar magnitudes

#5 - A - Dummies OK, hacemos una sola variable con varias modalidades (?)

#6 - A - En seleccion, probablemente sacar length y otras como country o language

#7 - A - Tal vez combiene poner x, x^2 y x^3 (?)