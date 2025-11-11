import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="Modelo Predictivo", page_icon="🍺")

st.title("Predictor de Calificación de Cerveza")
st.markdown("Predice la calificación general (*review_overall*) de una cerveza según sus características sensoriales.")

pipeline = joblib.load("beer_rating_model.pkl")

style_avg = None

try:
    DB_USER = "postgres"
    DB_PASS = "p1ensalo"
    DB_HOST = "localhost"
    DB_NAME = "beer_dw"
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

    query = """
    SELECT b.beer_style, AVG(f.review_overall) AS avg_overall
    FROM FactReview f
    JOIN DimBeer b ON f.beer_key = b.beer_key
    GROUP BY b.beer_style
    ORDER BY b.beer_style;
    """
    style_avg = pd.read_sql_query(query, engine)

except:
    if os.path.exists("style_avg.csv"):
        style_avg = pd.read_csv("style_avg.csv")
    else:
        st.error("No se encontró la base de datos ni el archivo 'style_avg.csv'.")
        st.stop()

style_mean = dict(zip(style_avg["beer_style"], style_avg["avg_overall"]))
global_mean = style_avg["avg_overall"].mean()

beer_style = st.selectbox("Estilo de la cerveza:", options=sorted(style_mean.keys()))
beer_abv = st.number_input("ABV (%)", min_value=0.0, max_value=20.0, value=5.0)
review_aroma = st.slider("Aroma (0-5)", 0.0, 5.0, 4.0)
review_taste = st.slider("Sabor (0-5)", 0.0, 5.0, 4.0)
review_palate = st.slider("Paladar (0-5)", 0.0, 5.0, 4.0)

if st.button(" Predecir calificación general"):
    style_te_value = style_mean.get(beer_style, global_mean)
    review_appearance = (review_aroma + review_taste + review_palate) / 3

    new_beer = pd.DataFrame([{
        "review_aroma": review_aroma,
        "review_appearance": review_appearance,
        "review_palate": review_palate,
        "review_taste": review_taste,
        "beer_abv": beer_abv,
        "style_te": style_te_value
    }])

    new_beer = new_beer[pipeline.named_steps['preprocessor'].feature_names_in_]
    predicted_overall = round(pipeline.predict(new_beer)[0], 2)

    st.success(f" Calificación estimada para '{beer_style}': **{predicted_overall} / 5.0**")



