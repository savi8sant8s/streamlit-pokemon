BASE_URL="https://gist.githubusercontent.com/leoyuholo/b12f882a92a25d43cf90e29812639eb3/raw/1abee7fb529dfacb374633f3a450b37634f8321a/pokemon.csv"

import pandas as pd
import streamlit as slt

df = pd.read_csv(BASE_URL)

slt.title("Machine Learning - Pokémon")

slt.subheader("5 primeiras linhas")
slt.write(df.head())

slt.subheader("Apagando primeira coluna que está sendo usada como index")
df = df.drop(df.columns[0], axis=1)
slt.write(df.head())

slt.subheader("Convertendo coluna is_legendary para booleano")
df["is_legendary"] = df["is_legendary"].astype("bool")
slt.write(df["is_legendary"])

slt.subheader("Checar se há valores nulos")
slt.write(df.isnull().sum())

