from typing import Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse,HTMLResponse
import pandas as pd
import ast
from fastapi.staticfiles import StaticFiles
import time
import asyncio

app = FastAPI()


df_reviews = pd.read_parquet("reviews_eda.parquet")
df_items = pd.read_parquet("items_eda.parquet")
df_games = pd.read_parquet("games_eda.parquet")

# Endpoint de la función PlayTimeGenre: Debe devolver año con mas horas jugadas para dicho género. Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}
@app.get('/PlayTimeGenre')
def PlayTimeGenre(genero : str):
    
    
     # Reemplazar los valores nulos en la columna 'genres' con una cadena vacía
    df_games['genres'].fillna('', inplace=True)

    # Filtrar el DataFrame para obtener solo las filas que contienen el género específico
    filtro_genero = df_games['genres'].str.contains(genero, case=False, na=False)
    df_filtrado = df_games[filtro_genero]

    # Agrupar por 'item_id' y sumar los valores de 'playtime_forever'
    playtiem_item = df_items.groupby('item_id')['playtime_forever'].sum().reset_index()

    df_merge = pd.merge(df_filtrado, playtiem_item, on='item_id', how='inner')

    df_agrupado = df_merge.groupby('year')['playtime_forever'].sum().reset_index()


    # Encontrar el año con la mayor suma de 'playtime_forever'
    df_agrupado['year'] = df_agrupado['year'].astype(int)
    anio_mayor_suma = df_agrupado.loc[df_agrupado['playtime_forever'].idxmax(), 'year']
    respuesta = {f"Año de lanzamiento con más horas jugadas para Género {genero}": anio_mayor_suma}
    return respuesta


@app.get('/UserForGenre')
def UserForGenre(genero: str):
    # Filtrar el DataFrame para el género especificado
    filtro_genero = df_games['genres'].str.contains(genero, case=False, na=False)
    df_filtrado = df_games[filtro_genero]

    # Realizar el merge con df_items
    df_merge = pd.merge(df_filtrado, df_items, on='item_id', how='inner')

    # Agrupar por 'user_id' y 'year', sumar las horas jugadas
    df_agrupado = df_merge.groupby(['user_id', 'year'])['playtime_forever'].sum().reset_index()

    # Encontrar el usuario con la máxima suma de horas jugadas
    idx_max_playtime = df_agrupado['playtime_forever'].idxmax()
    usuario_max_playtime = df_agrupado.loc[idx_max_playtime, 'user_id']

    # Filtrar el DataFrame para el usuario con máxima suma de horas jugadas
    df_usuario = df_agrupado[df_agrupado['user_id'] == usuario_max_playtime]

    # Crear el formato "Horas jugadas"
    resultado_final = [{'Año': int(row['year']), 'Horas': int(row['playtime_forever'])} for _, row in df_usuario.iterrows()]
    
    return {"Usuario con más horas jugadas para Género {}:".format(genero): usuario_max_playtime, "Horas jugadas": resultado_final}

@app.get('/UsersRecommend')
def UsersRecommend(año: int):
    # Filtrar el DataFrame para el año dado y las recomendaciones positivas
    df_reviews['posted_date'] = pd.to_datetime(df_reviews['posted_date'], errors='coerce')
    df_reviews['year_posted'] = df_reviews['posted_date'].dt.year
    df_2 = df_games[["item_id","title"]]
    df_filtrado = pd.merge(df_reviews, df_2, on='item_id', how='inner')
    df_filtrado = df_filtrado[(df_reviews['year_posted'] == año) & (df_reviews['recommend'] == True)]

    # Contar el número de recomendaciones para cada juego
    top_juegos = df_filtrado['title'].value_counts().head(3)

    # Crear el formato de salida
    resultado_final = [{"Puesto {}: {}".format(i+1, juego): recomendaciones} for i, (juego, recomendaciones) in enumerate(top_juegos.items())]

    return resultado_final


@app.get('/UsersWorstDeveloper')
def UsersWorstDeveloper(year : int):
    df_filter = df_reviews[["item_id","recommend","posted_date"]]
    df_info = df_games[["item_id","developer"]]
    df_worstDeveloper = pd.merge(df_filter,df_info, on ="item_id",how= "inner")
    df_worstDeveloper['posted_date'] = pd.to_datetime(df_reviews['posted_date'], errors='coerce')
    df_worstDeveloper['year_posted'] = df_reviews['posted_date'].dt.year
    df_worstDeveloper = df_worstDeveloper.drop(columns=['posted_date'])
    # Filtrar por el año deseado y recomendaciones negativas
    df_filtered = df_worstDeveloper[(df_worstDeveloper['year_posted'] == year) & (df_worstDeveloper['recommend'] == False)]

    # Contar la cantidad de juegos no recomendados por cada desarrolladora
    worst_developer_counts = df_filtered.groupby('developer')['item_id'].count().reset_index()

    # Ordenar por la cantidad de juegos no recomendados en orden descendente
    top_worst_developers = worst_developer_counts.sort_values(by='item_id', ascending=False).head(3)

    # Crear la lista de resultados en el formato deseado
    resultado = [{"Puesto {}".format(i+1): developer} for i, developer in enumerate(top_worst_developers['developer'])]

    return resultado


@app.get('/Sentiment_Analysis')
def sentiment_analysis(empresa_desarrolladora):
    df_10 = df_reviews[["item_id","sentiment_analysis"]]
    df_11 = df_games[["developer","item_id"]]
    df_12 = pd.merge(df_10,df_11, on = "item_id", how= "inner") 
    # Filtrar el DataFrame por la empresa desarrolladora especificada
    df_empresa = df_12[df_12['developer'] == empresa_desarrolladora]

    # Contar la cantidad de registros por análisis de sentimiento
    conteo_sentimientos = df_empresa['sentiment_analysis'].value_counts()

    # Crear el diccionario de resultados en el formato deseado
    resultado = {empresa_desarrolladora: {'Negative': conteo_sentimientos.get(0, 0),
                                           'Neutral': conteo_sentimientos.get(1, 0),
                                           'Positive': conteo_sentimientos.get(2, 0)}}

    return resultado
