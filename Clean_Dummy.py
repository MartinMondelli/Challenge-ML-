import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

def replace_na(df, col):
    df[col] = df[col].fillna(df[col].median())
    df[col] = df[col].clip(lower=0)
    return df[col]

def winsorize_features(df):
    df['budget'] = np.array(winsorize(df['budget'], limits=[0.05, 0.05]))
    df['length_wind'] = np.array(winsorize(df['length'], limits=[0.02, 0.02]))
    df['popularity_score_wind'] = np.array(winsorize(df['popularity_score'], limits=[0.01,0.01]))
    # Aplicamos margenes para llegar a 2% del total
    return df

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
  x_dum['log_popularity_score_scaled'] = np.log1p(df['popularity_score_scaled'])
  x_dum['log_popularity_score_scaled'] = x_dum['log_popularity_score_scaled'].replace(-np.inf, min(x_dum['log_popularity_score_scaled']))
  x_dum['length_2'] = x_dum['length_scaled'] ** 2
  x_dum['length_3'] = x_dum['length_scaled'] ** 3
  #Dummy para NA ===================================================================
  x_dum['NA'] = (pd.isna(df['length'])) & (pd.isna(df['popularity_score'])) & (pd.isna(df['budget']))
  return x_dum
