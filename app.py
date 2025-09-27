import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json

# Função para carregar os dados
def carregar_dados():
    url_movies = "https://drive.google.com/uc?export=download&id=17dWfqGAtdKZAR0rTCT6cv7weiIcrNycZ"
    url_credits = "https://drive.google.com/uc?export=download&id=1hQwFfz4ZXtF9UYwiH7VEwThbg207XjRL"

    movies_df = pd.read_csv(url_movies)
    credits_df = pd.read_csv(url_credits, engine='python', on_bad_lines='skip')
    df = movies_df.merge(credits_df, left_on="id", right_on="movie_id")
    df.drop("movie_id", axis=1, inplace=True)
    return df

# Função para tratar colunas JSON
def parse_json_column(df, column):
    for index, row in df.iterrows():
        try:
            df.at[index, column] = json.loads(row[column])
        except (TypeError, json.JSONDecodeError):
            df.at[index, column] = []
    return df

# Carregar os dados
df = carregar_dados()

# Parsing JSON columns
df = parse_json_column(df, 'genres')
df = parse_json_column(df, 'keywords')
df = parse_json_column(df, 'cast')
df = parse_json_column(df, 'crew')

# Exibir os primeiros dados
st.write("Primeiras Linhas do DataFrame:")
st.write(df.head())

# Visualização da distribuição das avaliações
st.write("Distribuição das Avaliações (Vote Average):")
sns.histplot(df['vote_average'], kde=True)
st.pyplot()

# Visualização do gráfico de dispersão entre orçamento e receita
st.write("Gráfico de Dispersão entre Orçamento e Receita:")
sns.scatterplot(data=df, x='budget', y='revenue')
st.pyplot()

# Função para obter o gênero principal
def main_genre_in_list(genre_list):
    if genre_list:
        return genre_list[0]['name']
    return None

# Criar a coluna de gênero principal
df['main_genre'] = df['genres'].apply(main_genre_in_list)

# Boxplot para avaliação por gênero
st.write("Boxplot das Avaliações por Gênero Principal:")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='main_genre', y='vote_average')
plt.xticks(rotation=90)
st.pyplot()

st.write("Contagem de Filmes por Idioma Original:")
plt.figure(figsize=(10, 8))
sns.countplot(data=df, y='original_language', order=df['original_language'].value_counts().index)
st.pyplot()

# Divisão em variáveis de entrada (X) e variável de saída (y)
X = df[['budget', 'revenue', 'popularity', 'vote_count']]
y = df['vote_average']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Exibir os resultados da regressão
st.write(f"MSE: {mse}")
st.write(f"RMSE: {rmse}")
st.write(f"R²: {r2}")

