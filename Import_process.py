import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Recover datos
df_train_in = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_train_features.csv",index_col=0)
y_train = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_train_revenue.csv",index_col=0)
df_test = pd.read_csv("https://edouardpauwels.fr/MLM2DSSS/challenge_test_features.csv",index_col=0)

# Scatter simple
plt.figure(figsize=(8,6))
sns.scatterplot(x=df_train_in["budget"], y=y_train["revenue"])
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.title("Scatter plot: Budget vs Revenue")
plt.show()
