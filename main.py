def carregar_dados():
    url_movies = "https://drive.google.com/uc?export=download&id=17dWfqGAtdKZAR0rTCT6cv7weiIcrNycZ"
    url_credits = "https://drive.google.com/uc?export=download&id=1hQwFfz4ZXtF9UYwiH7VEwThbg207XjRL"

    movies_df = pd.read_csv(url_movies)
    credits_df = pd.read_csv(url_credits, engine='python', on_bad_lines='skip')
    df = movies_df.merge(credits_df, left_on="id", right_on="movie_id")
    df.drop("movie_id", axis=1, inplace=True)
    return df

df = carregar_dados()
