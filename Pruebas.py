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


# Imprimir toda la fila
""""
## Display basic information about the data
df_train_in.info()
y_train.info()
df_test.info()

df_train_in.head()

def clean_dummies (df):
  x_dum = df.copy()
  #Separating year and month =======================================================
  date_split = df["date"].str.split("/", expand=True)
  x_dum["month"] = date_split[0].astype(int)
  x_dum["year"] = date_split[2].astype(int)
  #Dummies
  # Sequels ========================================================================
  x_dum["sequels"] = df["collection"].apply(lambda c: 0 if pd.isna(c) else 1)
  print(x_dum["sequels"].value_counts())
  #1600 NO tienen secuelas contra 400 que SI
  # Seasons and gender correlation =================================================
  x_dum["season_horror"] = x_dum["month"].apply(lambda m: 1 if m in [10, 11] else 0)
  x_dum["season_romance"] = x_dum["month"].apply(lambda m: 1 if m == 2 else 0)
  x_dum["season_family"] = x_dum["month"].apply(lambda m: 1 if m in [12, 1] else 0)
  print(x_dum["season_horror"].value_counts())
  #1650 NO fueron lanzadas en Halloween 450 SI (NO hablamos aun de peliculas de terror en Halloween)
  print(x_dum["season_romance"].value_counts())
  #1850 NO fueron lanzadas en San Valentin 150 SI (NO hablamos aun de peliculas de romance en San Valentin)
  print(x_dum["season_family"].value_counts())
  #1700 NO fueron lanzadas en Navidad 300 SI (NO hablamos aun de peliculas de familia en Navidad)
  # Top Actor ======================================================================
  main_actors = [
    "Marlon Brando", "Humphrey Bogart", "James Stewart", "Gary Cooper", "Bette Davis",
    "Katharine Hepburn", "Audrey Hepburn", "Jack Nicholson", "Tom Hanks", "Meryl Streep",
    "Al Pacino", "Robert De Niro", "Daniel Day-Lewis", "Clint Eastwood", "Dustin Hoffman",
    "Jack Lemmon", "Paul Newman", "Shirley MacLaine", "Sidney Poitier", "Ingrid Bergman",
    "Elizabeth Taylor", "Greta Garbo", "Judy Garland", "Marilyn Monroe", "Cary Grant",
    "Charlie Chaplin", "Laurence Olivier", "Henry Fonda", "Gene Kelly", "Harrison Ford",
    "Clark Gable", "Tom Cruise", "Jodie Foster", "Nicole Kidman", "Morgan Freeman",
    "Denzel Washington", "Leonardo DiCaprio", "Cate Blanchett", "Sylvester Stallone",
    "Sandra Bullock", "Julia Roberts", "Emma Thompson", "Helen Mirren", "Brad Pitt",
    "George Clooney", "Angelina Jolie", "Matt Damon", "Robert Downey Jr.", "Chris Hemsworth",
    "Chris Evans", "Scarlett Johansson", "Anne Hathaway", "Hugh Jackman", "Johnny Depp",
    "Maggie Smith", "Ian McKellen", "Christopher Lee", "Samuel L. Jackson", "Morgan Freeman",
    "Michael Caine", "Alfred Hitchcock", "Marion Cotillard", "Ryan Gosling", "Kate Winslet",
    "Keira Knightley", "Sean Connery", "Michael Douglas", "Sigourney Weaver", "Anthony Hopkins",
    "Gary Oldman", "Emma Watson", "Daniel Radcliffe", "Robert Pattinson", "Matt LeBlanc",
    "Jennifer Lawrence", "Chris Pratt", "Chris Pine", "Gal Gadot", "Zoe Saldana"
]
  x_dum["star"] = df["cast"].apply(
    lambda cast: 1 if isinstance(cast, str) and any(actor in main_actors for actor in cast.split(",")) else 0
)
  print(x_dum["star"].value_counts())
  #1550 NO tienen grandes actores 450 SI
  # Big company (only Top 10) =======================================================
  big_companies = ["Twentieth Century Fox Film Corporation", "Universal Pictures", "Warner Bros. Pictures",
                   "Columbia Pictures Corporation", "Walt Disney Pictures", "Marvel Studios", "Paramount Pictures",
                   "Legendary Pictures", "New Line Cinema", "DreamWorks Animation"]
  x_dum["big_comp"] = df["company"].apply(
      lambda company: 1 if isinstance(company, str) and any(company in big_companies for company in company.split(",")) else 0
  )
  print(x_dum["big_comp"].value_counts())
  #1550 NO son de grandes productoras 450 SI
  #Director lo podemos hacer con crew, util?? =======================================
  return x_dum
df_train_in_2 = clean_dummies(df_train_in)
df_test_processed = clean_dummies(df_test)

df_train_run = df_train_in_2[["sequels", "star", "season_horror", "season_romance", "season_family",
                            "star", "big_comp"]]
df_test_run = df_test_processed[["sequels", "star", "season_horror", "season_romance", "season_family",
                            "star", "big_comp"]]

param_clfrf = {
    'max_depth':[8,10,20,25,34,40,45]     # Max deep of each tree
}

clfrf = RandomForestRegressor()
clfrf_cv = GridSearchCV(
    estimator=clfrf,
    param_grid=param_clfrf,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Optimizar por precisión
    verbose=1  # Mostrar progreso
)
y_output = clfrf_cv.fit(df_train_run, y_train).predict(df_test_run)
"""

########################################################################
########################################################################
############ Codigo que funciona de recuperacion #######################
########################################################################
########################################################################
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV # for cross-validation
import numpy as np
from functools import reduce
import sklearn as sk
from sklearn.preprocessing import StandardScaler


# Para pullear : shift + command + p

## Import data as a panda dataframe
"""
df_train_in = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_train_features.csv",index_col=0)
y_train = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_train_revenue.csv",index_col=0)
df_test = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_test_features.csv",index_col=0)

## Display basic information about the data
df_train_in.info()
y_train.info()
df_test.info()

df_train_in.head()
"""

#Scalling
def scale_features(train, test, cols):
    train_scaled = train.copy()
    test_scaled = test.copy()

    for col in cols:
        scaler = StandardScaler()
        # Fit en train, transform en ambos
        train_scaled[col + "_scaled"] = scaler.fit_transform(train[[col]])
        test_scaled[col + "_scaled"] = scaler.transform(test[[col]])

    return train_scaled, test_scaled

def clean_dummies (df):
  x_dum = df.copy()
  #Separating year and month =====================================================
  date_split = df["date"].str.split("/", expand=True)
  x_dum["month"] = date_split[0].astype(int)
  x_dum["year"] = date_split[2].astype(int)
  #Dummies
  # Sequels ======================================================================
  x_dum["sequels"] = df["collection"].apply(lambda c: 0 if pd.isna(c) else 1)
  #print(x_dum["sequels"].value_counts())
  #1600 NO tienen secuelas contra 400 que SI
  # Seasons and gender correlation ===============================================
  x_dum["season_horror"] = x_dum["month"].apply(lambda m: 1 if m in [10, 11] else 0)
  x_dum["season_romance"] = x_dum["month"].apply(lambda m: 1 if m == 2 else 0)
  x_dum["season_family"] = x_dum["month"].apply(lambda m: 1 if m in [12, 1] else 0)
  #print(x_dum["season_horror"].value_counts())
  #1650 NO fueron lanzadas en Halloween 450 SI (NO hablamos aun de peliculas de terror en Halloween)
  #print(x_dum["season_romance"].value_counts())
  #1850 NO fueron lanzadas en San Valentin 150 SI (NO hablamos aun de peliculas de romance en San Valentin)
  #print(x_dum["season_family"].value_counts())
  #1700 NO fueron lanzadas en Navidad 300 SI (NO hablamos aun de peliculas de familia en Navidad)
  # Top Actor ====================================================================
  main_actors = [
    "Marlon Brando", "Humphrey Bogart", "James Stewart", "Gary Cooper", "Bette Davis",
    "Katharine Hepburn", "Audrey Hepburn", "Jack Nicholson", "Tom Hanks", "Meryl Streep",
    "Al Pacino", "Robert De Niro", "Daniel Day-Lewis", "Clint Eastwood", "Dustin Hoffman",
    "Jack Lemmon", "Paul Newman", "Shirley MacLaine", "Sidney Poitier", "Ingrid Bergman",
    "Elizabeth Taylor", "Greta Garbo", "Judy Garland", "Marilyn Monroe", "Cary Grant",
    "Charlie Chaplin", "Laurence Olivier", "Henry Fonda", "Gene Kelly", "Harrison Ford",
    "Clark Gable", "Tom Cruise", "Jodie Foster", "Nicole Kidman", "Morgan Freeman",
    "Denzel Washington", "Leonardo DiCaprio", "Cate Blanchett", "Sylvester Stallone",
    "Sandra Bullock", "Julia Roberts", "Emma Thompson", "Helen Mirren", "Brad Pitt",
    "George Clooney", "Angelina Jolie", "Matt Damon", "Robert Downey Jr.", "Chris Hemsworth",
    "Chris Evans", "Scarlett Johansson", "Anne Hathaway", "Hugh Jackman", "Johnny Depp",
    "Maggie Smith", "Ian McKellen", "Christopher Lee", "Samuel L. Jackson", "Morgan Freeman",
    "Michael Caine", "Alfred Hitchcock", "Marion Cotillard", "Ryan Gosling", "Kate Winslet",
    "Keira Knightley", "Sean Connery", "Michael Douglas", "Sigourney Weaver", "Anthony Hopkins",
    "Gary Oldman", "Emma Watson", "Daniel Radcliffe", "Robert Pattinson", "Matt LeBlanc",
    "Jennifer Lawrence", "Chris Pratt", "Chris Pine", "Gal Gadot", "Zoe Saldana"
]
  x_dum["star"] = df["cast"].apply(
    lambda cast: 1 if isinstance(cast, str) and any(actor in main_actors for actor in cast.split(",")) else 0
)
  #print(x_dum["star"].value_counts())
  #1550 NO tienen grandes actores 450 SI
  # Big company (only Top 10) =====================================================
  big_companies = ["Twentieth Century Fox Film Corporation", "Universal Pictures", "Warner Bros. Pictures",
                   "Columbia Pictures Corporation", "Walt Disney Pictures", "Marvel Studios", "Paramount Pictures",
                   "Legendary Pictures", "New Line Cinema", "DreamWorks Animation"]
  x_dum["big_comp"] = df["company"].apply(
      lambda company: 1 if isinstance(company, str) and any(company in big_companies for company in company.split(",")) else 0
  )
  #print(x_dum["big_comp"].value_counts())
  #1550 NO son de grandes productoras 450 SI
  #Director lo podemos hacer con crew, util?? =====================================
  #Dummy for decades - CORREGIDO: usar operaciones vectorizadas y años completos
  # Crear variables dummy para cada década usando operaciones vectorizadas
  x_dum['decade_1920'] = ((x_dum['year'] >= 1920) & (x_dum['year'] < 1930)).astype(int)
  x_dum['decade_1930'] = ((x_dum['year'] >= 1930) & (x_dum['year'] < 1940)).astype(int)
  x_dum['decade_1940'] = ((x_dum['year'] >= 1940) & (x_dum['year'] < 1950)).astype(int)
  x_dum['decade_1950'] = ((x_dum['year'] >= 1950) & (x_dum['year'] < 1960)).astype(int)
  x_dum['decade_1960'] = ((x_dum['year'] >= 1960) & (x_dum['year'] < 1970)).astype(int)
  x_dum['decade_1970'] = ((x_dum['year'] >= 1970) & (x_dum['year'] < 1980)).astype(int)
  x_dum['decade_1980'] = ((x_dum['year'] >= 1980) & (x_dum['year'] < 1990)).astype(int)
  x_dum['decade_1990'] = ((x_dum['year'] >= 1990) & (x_dum['year'] < 2000)).astype(int)
  x_dum['decade_2000'] = ((x_dum['year'] >= 2000) & (x_dum['year'] < 2010)).astype(int)
  x_dum['decade_2010'] = ((x_dum['year'] >= 2010) & (x_dum['year'] < 2020)).astype(int)
  x_dum['decade_2020'] = ((x_dum['year'] >= 2020) & (x_dum['year'] < 2030)).astype(int)
  # Dummy for marketing ============================================================
  x_dum["marketing_web"] = x_dum["webpage"].apply(lambda m: 0 if pd.isna(m) else 1)
  # Dummy for en ===================================================================
  x_dum["english"] = x_dum["language"].apply(lambda m: 0 if m=="en" else 1)
  # Dummy for country = ============================================================
  x_dum["US"] = x_dum["country"].apply(lambda m: 0 if m=="US" else 1)

  # Rescaling ======================================================================
  x_dum['log_budget'] = np.log(df['budget'].replace(0, np.nan))
  x_dum['log_budget'] = x_dum['log_budget'].replace([-np.inf, np.inf], -1).fillna(-1)
  x_dum['popularity_score'] = df['popularity_score'].fillna(
      np.min(-1))  # We tried with this "np.min(df['popularity_score'])"
  x_dum['length'] = df['length'].fillna(-1)  # We tried with this "np.mean(df['length'])"
  #Dummy para NA ===================================================================
  x_dum['NA'] = (pd.isna(df['length'])) & (pd.isna(df['popularity_score'])) & (pd.isna(df['budget']))
  return x_dum

"""
df_train_in_2 = clean_dummies(df_train_in)
df_test_processed = clean_dummies(df_test)

df_train_run = df_train_in_2[["sequels", "star", "season_horror", "season_romance", "season_family",
                            "big_comp", "log_budget", "popularity_score", "length", "NA"]]
df_test_run = df_test_processed[["sequels", "star", "season_horror", "season_romance", "season_family",
                            "big_comp", "log_budget", "popularity_score", "length", "NA"]]

# sin esto obtengo NaN para el test error: "season_horror", "season_romance", "season_family"
param_clfgb = {
    'learning_rate': [0.05,0.1,0.2],
    'n_estimators' : [100,200,300,400,500]
}

gbr = GradientBoostingRegressor()

param_gbr = {
    "n_estimators": [500],
    "max_depth": [5],
    "learning_rate": [0.01]
}

gbr_cv = GridSearchCV(
    estimator=gbr,
    param_grid=param_gbr,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',  # métrica de regresión
    verbose=1
)

y_output = gbr_cv.fit(df_train_run, y_train.values.ravel()).predict(df_test_run)

# Redondear a la decena más cercana
strRes = [str(s) for s in np.round(y_output, -1)]

predStr = reduce(lambda x, y: x + ', ' + y, strRes)

print(predStr)
"""