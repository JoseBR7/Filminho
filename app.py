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
@st.cache_data
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

# Função para obter o gênero principal
def main_genre_in_list(genre_list):
    if genre_list:
        return genre_list[0]['name']
    return None

# Criar a coluna de gênero principal
df['main_genre'] = df['genres'].apply(main_genre_in_list)

# Limpeza de valores nulos ou vazios na coluna 'main_genre'
df['main_genre'] = df['main_genre'].fillna('Desconhecido')  # Substituir NaN por 'Desconhecido'
df['main_genre'] = df['main_genre'].replace('', 'Desconhecido')  # Substituir valores vazios por 'Desconhecido'

# Definindo o layout
st.set_page_config(page_title="Análise de Filmes", page_icon="🎬", layout="wide")

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

# Função para gerar gráficos
def plot_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['vote_average'], kde=True, color='purple', ax=ax)
    ax.set_title("Distribuição das Avaliações dos Filmes")
    st.pyplot(fig)

def plot_scatter():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='budget', y='revenue', color='teal', ax=ax)
    ax.set_title("Relação entre Orçamento e Receita dos Filmes")
    ax.set_xlabel("Orçamento")
    ax.set_ylabel("Receita")
    st.pyplot(fig)

def plot_boxplot():
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='main_genre', y='vote_average', palette='Set2', ax=ax)
    ax.set_title("Avaliações por Gênero de Filme")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

def plot_countplot():
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(data=df, y='original_language', order=df['original_language'].value_counts().index, palette='coolwarm', ax=ax)
    ax.set_title("Distribuição de Filmes por Idioma Original")
    st.pyplot(fig)

# Adicionando uma barra lateral
st.sidebar.header("Opções de Visualização")
sidebar_option = st.sidebar.selectbox(
    "Escolha uma visualização:",
    ["Distribuição das Avaliações", "Gráfico de Dispersão", "Boxplot por Gênero", "Contagem de Idioma", "Sugestões de Negócio"]
)

# Renderizar gráfico conforme a seleção do usuário
if sidebar_option == "Distribuição das Avaliações":
    # Cabeçalho e descrição apenas nesta aba
    st.title("Análise de Filmes e Previsão de Avaliações")
    st.markdown("""
        Bem-vindo à análise de dados dos filmes! Aqui, você pode explorar a distribuição das avaliações, 
        os gráficos de orçamento versus receita, os gêneros de filmes, e também testar um modelo de regressão 
        linear para prever as avaliações dos filmes com base em características como orçamento e popularidade.
    """)

    # Exibir os primeiros dados com um título
    st.subheader("Primeiras Linhas do DataFrame")
    st.dataframe(df[["budget", "revenue", "vote_average", "popularity", "vote_count"]].describe())

    # Exibir os resultados da regressão linear
    st.subheader("Resultados da Regressão Linear")
    st.write(f"**MSE (Erro Quadrático Médio)**: {mse:.2f}")
    st.write(f"**RMSE (Raiz do Erro Quadrático Médio)**: {rmse:.2f}")
    st.write(f"**R² (Coeficiente de Determinação)**: {r2:.2f}")

    # Exibição dos coeficientes do modelo
    st.write("**Coeficientes do Modelo de Regressão Linear**:")
    coef_df = pd.DataFrame({
        'Variáveis': X.columns,
        'Coeficientes': model.coef_
    })
    st.write(coef_df)

    # Gráfico de distribuição
    st.subheader("Distribuição das Avaliações (Vote Average)")
    plot_distribution()

elif sidebar_option == "Gráfico de Dispersão":
    st.subheader("Gráfico de Dispersão entre Orçamento e Receita")
    plot_scatter()

elif sidebar_option == "Boxplot por Gênero":
    st.subheader("Boxplot das Avaliações por Gênero Principal")
    plot_boxplot()

elif sidebar_option == "Contagem de Idioma":
    st.subheader("Contagem de Filmes por Idioma Original")
    plot_countplot()

elif sidebar_option == "Sugestões de Negócio":
    # Sugestões de Negócio
    st.title("Sugestões de Negócio")
    st.markdown("""
        Com base nas análises realizadas, podemos fazer algumas sugestões que podem ser úteis para o setor de cinema:

        ### 1. Otimização de Orçamento e Receita
        - A análise indicou que há uma forte correlação entre o **orçamento** e a **receita** dos filmes. 
        - Sugestão: Investir de maneira mais estratégica em filmes de maior orçamento, focando em marketing eficaz e um bom planejamento de distribuição.

        ### 2. Exploração de Gêneros Populares
        - Certos gêneros de filmes, como **Ação**, **Aventura** e **Ficção Científica**, tendem a gerar boas avaliações e popularidade.
        - Sugestão: Priorizar a produção desses gêneros ou investir em inovações dentro desses segmentos, como combinações com outros gêneros ou experimentações.

        ### 3. Foco na Popularidade e Marketing
        - Filmes com maior **popularidade** têm melhores chances de obter boas avaliações.
        - Sugestão: Priorizar campanhas de marketing para aumentar a popularidade de filmes antes do lançamento, utilizando redes sociais e influenciadores, além de apostar em pré-estreias e engajamento com a audiência.

        ### 4. Idioma Original e Mercado Internacional
        - Filmes com uma maior diversidade de **idiomas originais** tendem a atrair um público mais global.
        - Sugestão: Investir em versões dubladas e legendadas para diferentes mercados, além de considerar produções que atendam a uma diversidade de culturas e idiomas.

        ### 5. Análise de Dados e Previsões
        - O modelo de regressão linear mostrou que podemos prever as avaliações de filmes com base em **orçamento**, **popularidade** e **número de votos**.
        - Sugestão: Utilizar esses modelos preditivos para tomar decisões sobre onde investir em produções, quais filmes têm maior potencial de sucesso, e realizar ajustes em tempo real para otimizar as receitas.

        ## Considerações Finais
        As análises e modelos preditivos podem ser uma ferramenta poderosa para os tomadores de decisão na indústria cinematográfica. Usando essas informações, é possível otimizar investimentos, selecionar os melhores gêneros e ajustar estratégias de marketing para garantir o sucesso de futuros lançamentos.
    """)
