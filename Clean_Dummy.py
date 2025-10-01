import pandas as pd
#test
## Import data as a panda dataframe
df_train_in = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_train_features.csv",index_col=0)
y_train = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_train_revenue.csv",index_col=0)
df_test = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_test_features.csv",index_col=0)

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