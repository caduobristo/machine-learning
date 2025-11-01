# %%

import pandas as pd

df = pd.read_csv('data/abt_churn.csv')

# %%

oot = df[df['dtRef']==df['dtRef'].max()].copy()
oot

# %%

df_train =df[df['dtRef']<df['dtRef'].max()].copy()
df_train

# %%

features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df[features], df[target]

# %%

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=y)
print(y.mean())
print(y_train.mean())
print(y_test.mean())