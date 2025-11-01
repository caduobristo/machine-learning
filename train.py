# %%

import pandas as pd
pd.set_option('display.max_rows', None)

df = pd.read_csv('data/abt_churn.csv')

# %%

# out of time
oot = df[df['dtRef']==df['dtRef'].max()].copy()
print(oot)

'''
Dados coletados na úlima sazonalidade (nesse caso, último mês) para validação
posterior do modelo no tempo.
'''

# %%

df_train =df[df['dtRef']<df['dtRef'].max()].copy()
print(df_train)

'''
Dados que seram utilizados para treinar o modelo
'''

# %%

features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df[features], df[target]

'''
Separação das features e targets
'''

# %%

# SAMPLE

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=y)
print(y.mean())
print(y_train.mean())
print(y_test.mean())

'''
80% dos dados serão utilizados para o treino do modelo e os outros 20% serão para teste
A variável de target foi estratificada para que a distribuição fique uniforme entre o treino
e os teste.
'''

# %%

# EXPLORE (MISSING)

print(X_train.isna().sum().sort_values(ascending=False))

'''
Análise de dados faltantes.
'''

# %%

df_analise = X_train.copy()
df_analise[target] = y_train
# Média e mediana de cada variável em cada flag do target
summario = df_analise.groupby(by=target).agg(['mean', 'median']).T
# Diferença absoluta
summario['diff_abs'] = summario[0] - summario[1]
# Diferenã relativa
summario['diff_rel'] = summario[0]/summario[1]
summario = summario.sort_values(by='diff_rel', ascending=False)
print(summario)

'''
Análise da base de dados de treino para entender as variáveis que mais influenciam no target.
'''

# %%

from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

#plt.figure(dpi=700, figsize=[4, 4])
#tree.plot_tree(arvore, feature_names=X_train.columns,
#               filled=True, class_names=[str(i) for i in arvore.classes_])

# Serie com a importância de cada feature definido pela árvore
feature_importance = (pd.Series(arvore.feature_importances_, index=X_train.columns)
                       .sort_values(ascending=False)
                       .reset_index())
# Soma acumulativa da importãncia de cada feature
feature_importance['acum.']=feature_importance[0].cumsum()
# Separa apenas as features que, somadas, tem importãncia de até 96%
feature_importance[feature_importance['acum.'] < 0.96]
print(feature_importance)

'''
Árvore de classifição para entender melhor a importância de cada variável.
'''