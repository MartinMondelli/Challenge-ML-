from Clean_Dummy import clean_dummies, scale_features, winsorize_features, replace_na
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV # for cross-validation
import numpy as np
from functools import reduce

# Para pullear : shift + command + p

########################################################################
########################################################################

#Lanzar despues de Clean_Dummy.py

#Recover datos
df_train_in = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_train_features.csv",index_col=0)
y_train = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_train_revenue.csv",index_col=0)
df_test = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_test_features.csv",index_col=0)

#Replace NAs
df_train_in["length"] = replace_na(df_train_in, "length")
df_test["length"] = replace_na(df_test, "length")
df_train_in["popularity_score"] = replace_na(df_train_in, "popularity_score")
df_test["popularity_score"] = replace_na(df_test, "popularity_score")
df_train_in["budget"] = replace_na(df_train_in, "budget")
df_test["budget"] = replace_na(df_test, "budget")

#Winsorize
df_train_in = winsorize_features(df_train_in)
df_test = winsorize_features(df_test)

#Scale
df_train_in_scaled, df_test_scaled = scale_features(df_train_in, df_test, ["popularity_score", "length", "budget"])

#Get dummies
df_train_in_run = clean_dummies(df_train_in_scaled)
df_test_run = clean_dummies(df_test_scaled)

#Choose variables
features = ["sequels", "season_horror", "season_romance", "season_family", "big_comp", "budget_scaled",
            "log_popularity_score_scaled", "length_3", "NA"]
df_train_run = df_train_in_run[features]
df_test_run = df_test_run[features]

#Notas:
#["sequels", "season_horror", "season_romance", "season_family", "big_comp", "budget_scaled",
#            "log_popularity_score_scaled", "length_3", "NA"] = 1.696870
# "season_horror", "season_romance", "season_family" sin esto obtengo NaN para el test error (o sea muy grande)
#Con "log_budget", "popularity_score", "length" obtengo 1.88

########################################################################
# Modelo 1 GradientBoostingRegressor()
########################################################################
#Initialize the model
gbr = GradientBoostingRegressor()

#We use GridSearchCV to get the best parameters
param_gbr = {
    "n_estimators": [850],
    "max_depth": [3],
    "learning_rate": [0.01]
}
#Con n_estimators >= 800 obtengo NaN para el error

#We can use this scoring system: msle_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
gbr_cv = GridSearchCV(
    estimator=gbr,
    param_grid=param_gbr,
    cv=2,  # 5-fold cross-validation (cambiar cuando usemos el gridsearch de verdad)
    scoring="neg_mean_absolute_error",  # métrica de regresión
    verbose=1
)

gbr_cv_fit = gbr_cv.fit(df_train_run, y_train.values.ravel())
print(gbr_cv.best_params_)
#Mejor: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 850}

y_output = gbr_cv_fit.predict(df_test_run)
# Redondear la unidad mas cercana
strRes = [str(s) for s in np.round(y_output, 0)]

#Poner en forma del vector del profesor
predStr = reduce(lambda x, y: x + ', ' + y, strRes)

print(predStr)
"""

########################################################################
# Modelo 2
########################################################################
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
lgb_model.fit(df_train_run, y_train.values.ravel())
y_output = lgb_model.predict(df_test_run)

# Redondear la unidad mas cercana
strRes = [str(s) for s in np.round(y_output, 0)]

#Poner en forma del vector del profesor
predStr = reduce(lambda x, y: x + ', ' + y, strRes)

print(predStr)
"""
#Top resultados:
#1.763 : "sequels", "season_horror", "season_romance", "season_family", "big_comp", "english", "log_budget", "popularity_score", "length"