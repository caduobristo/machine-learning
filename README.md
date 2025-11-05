# Estudo de ML

Repositório dedicado aos meus estudos de *Machine Learning*, realizados a partir do curso do [Téo Me Why](https://www.youtube.com/@teomewhy). Aqui, estou acompanhando o curso, entendendo como o projeto é desenvolvido e depois refazendo tudo do meu jeito - aplicando o que aprendi, adicionando comentários e explorando as ideias com base no meu próprio conhecimento.

## Projeto: Churn

O objetivo do projeto é prever quais usuários deixam de acompanhar as lives do Téo.
- Base de dados: [Analytical Base Table Churn](https://www.kaggle.com/datasets/teocalvo/analytical-base-table-churn)

## Etapas do SEMMA

### Sample

- Leitura da base de dados `abt_churn.csv`.
- Separação dos dados em:
- Treino (df_train): utilizado para desenvolvimento e ajuste do modelo.
    - Out of Time (oot): dados do último período, reservados para validação futura.
    - Definição das features e do target (flagChurn).
- Divisão dos dados entre treino e teste (80/20) com estratificação do target para manter a proporção das classes.

### Explore

- Verificação de valores ausentes nas variáveis de treino.
- Criação de um resumo estatístico (média e mediana) das variáveis para cada classe do target.
- Cálculo das diferenças absolutas e relativas entre as classes, para identificar quais variáveis mais influenciam o churn.
- Treinamento de uma árvore de decisão simples apenas para analisar a importância das variáveis, sem foco em performance ainda.

## Modify

- Filtro para separar features que não são uma proporção.
- Discretização das features que não são proporção.
- Aplicação de OneHotEncoder nas features discretizadas (resultados levemente melhores obtidos).

## Model

- Criação de um pipeline com as modificações feitas nos dados e um modelo.
- Aplicação dos modelos de Regressão Logística, RandomForest e NaiveBayes.
- Utilização do MLFlow para ajudar na comparação dos resultados de cada modelo.