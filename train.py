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

X, y = df_train[features], df_train[target]

'''
Separação das features e targets
'''

# %%

X_oot = oot[features]
y_oot = oot[target]

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

print(feature_importance[feature_importance['acum.'] < 0.96])

'''
Árvore de classifição para entender melhor a importância de cada variável.
'''

# %%

best_features = (feature_importance[feature_importance['acum.'] < 0.96]['index']
                 .tolist())
X_train_best = X_train[best_features].copy()

'''
Utilizando apenas as melhores features na base de treino
'''

# %%

# MODIFY

from feature_engine import discretisation, encoding

# Filtro das features que não são uma proporção
features_discre = X_train_best.loc[:, ~X_train_best.columns.str.startswith('prop')].columns.to_list()

tree_discretisation = discretisation.DecisionTreeDiscretiser(variables=features_discre,
                                                             regression=False,
                                                             bin_output='bin_number',
                                                             cv=3)

onehot = encoding.OneHotEncoder(variables=features_discre, ignore_format=True)

'''
Discretização das variáveis que não são uma proporção utilizando uma árvore e aplicação de 
OnehotEncoder nas features discretizadas.
'''

# %%

# MODEL
from sklearn import linear_model, ensemble, naive_bayes, pipeline

#model = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=1000000)
#model = naive_bayes.GaussianNB()
model = ensemble.RandomForestClassifier(random_state=42, min_samples_leaf=20, n_jobs=-1, n_estimators=500)

model_pipeline = pipeline.Pipeline(
    steps=[
        ('Discretisation', tree_discretisation),
        ('Onehot', onehot),
        ('Reg', model)
    ]
)

'''
Criação de um pipeline com todas as tranformações feitas nos dados ecom vários modelos.
'''

# %%

from sklearn import metrics
import mlflow

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment(experiment_id=897907439885099594)

with mlflow.start_run(run_name=model.__str__()):
    mlflow.sklearn.autolog()

    model_pipeline.fit(X_train_best, y_train)

    y_train_predict = model_pipeline.predict(X_train_best)
    y_train_prob = model_pipeline.predict_proba(X_train_best)[:,1]

    train_acc = metrics.accuracy_score(y_train, y_train_predict)
    train_auc = metrics.roc_auc_score(y_train, y_train_prob)
    print("Treino acc:", train_acc)
    print("Treino auc:", train_auc)

    y_test_predict = model_pipeline.predict(X_test[best_features])
    y_test_prob = model_pipeline.predict_proba(X_test[best_features])[:,1]

    test_acc = metrics.accuracy_score(y_test, y_test_predict)
    test_auc = metrics.roc_auc_score(y_test, y_test_prob)
    print("Teste acc:", test_acc)
    print("Teste auc:", test_auc)

    y_ott_predict = model_pipeline.predict(X_oot[best_features])
    y_ott_prob = model_pipeline.predict_proba(X_oot[best_features])[:,1]

    oot_acc = metrics.accuracy_score(y_oot, y_ott_predict)
    oot_auc = metrics.roc_auc_score(y_oot, y_ott_prob)
    print("oot acc:", oot_acc)
    print("oot auc:", oot_auc)

    mlflow.log_metrics({
        "train_acc": train_acc,
        "train_auc": train_auc,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "oot_acc": oot_acc,
        "oot_auc": oot_auc
    })

'''
Predição das bases de treino, teste e out of time para validação dos resultados obtidos pela regressão.
Utilização do mlflow para facilitar a comparação de resultados.
'''