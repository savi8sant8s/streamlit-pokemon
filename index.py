BASE_URL="https://gist.githubusercontent.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/raw/92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv"

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import streamlit as slt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from category_encoders import OrdinalEncoder

df = pd.read_csv(BASE_URL)

slt.title("Machine Learning - Pokémons")

slt.subheader("Apresentando 5 primeiras linhas")
slt.write(df.head())

slt.subheader("Apresentando resumo estatístico")
slt.write(df.describe())

slt.subheader("Apagando primeira coluna que está sendo usada como index")
df = df.drop(df.columns[0], axis=1)
slt.write(df.head())

slt.subheader("Convertendo coluna Legendary para booleano numérico")
df["Legendary"] = df["Legendary"].map({False: 0, True: 1})
slt.write(df.head())

slt.subheader("Checando se há valores nulos")
slt.write(df.isnull().sum())

slt.subheader("Apresentando gráfico de correlação")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), ax=ax)
slt.write(fig)

slt.subheader("Removendo coluna de Generation que tem baixa correlação com a coluna Legendary")
df = df.drop(["Generation"], axis=1)
slt.write(df.head())

slt.subheader("Removendo coluna de Name que não é relevante para o modelo")
df = df.drop(["Name"], axis=1)
slt.write(df.head())

slt.subheader("Apresentando histograma para o Type1 de pokémons")
fig, ax = plt.subplots(figsize=(15, 2))
sns.countplot(x="Type 1", data=df, ax=ax)
slt.write(fig)

slt.subheader("Convertendo colunas Type1 e Type2 para categorias numéricas")
encoder = OrdinalEncoder()
df["Type 1"] = encoder.fit_transform(df["Type 1"])
df["Type 2"] = encoder.fit_transform(df["Type 2"])
slt.write(df)

slt.subheader("Separando dados de treino e teste")
def split_train_test(data, test_ratio):
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data)*test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(df, 0.2)

slt.write("Tamanho do dados de treino: " + str(len(train_set)))
slt.write("Tamanho do dados de teste: " + str(len(test_set)))

train_x = train_set.drop("Legendary", axis=1)
train_y = train_set["Legendary"].copy()

test_x = test_set.drop("Legendary", axis=1)
test_y = test_set["Legendary"].copy()

slt.subheader("Aplicando modelo de SVM com Cross Validation")
model = SVC()
scores = cross_val_score(model, train_x, train_y, cv=10, scoring="neg_mean_squared_error")
rmse_scores = np.sqrt(-scores)
slt.write("Scores: " + str(rmse_scores))
slt.write("Média: " + str(rmse_scores.mean()))
slt.write("Desvio padrão: " + str(rmse_scores.std()))

slt.subheader("Aplicando GridSearch para encontrar os melhores parâmetros para o modelo de SVM")
param_grid = [
  {'kernel': ['linear', 'rbf']}
]
model = SVC()
grid_search = GridSearchCV(model, param_grid, cv=10, scoring="neg_mean_squared_error")
grid_search.fit(train_x, train_y)
slt.write("Melhor kernel: " + str(grid_search.best_params_))