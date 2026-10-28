import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="Olist Delivery Delay Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Caricamento del modello salvato
@st.cache_resource
def load_model():
    data = joblib.load("models/xgb_model.joblib")
    return data["model"], data["features"]


model, feature_names = load_model()

# --- HEADER & TITOLO ---
st.title("🚚 Olist E-Commerce Brazil - Delay Risk Predictor")
st.markdown(
    """
    Questa applicazione utilizza un modello di **Machine Learning (XGBoost)** per stimare la probabilità 
    che un ordine e-commerce subisca un **ritardo nella consegna** al momento del checkout.
    """
)

st.divider()

# --- METRICHE DI BUSINESS GENERALI (KPIs) ---
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
with col_kpi1:
    st.metric(label="Tasso Medio di Ritardo", value="6.66%")
with col_kpi2:
    st.metric(
        label="Rating Medio (In Orario vs Ritardo)",
        value="4.29 ⭐ ➔ 2.27 ⭐",
        delta="-47% Gradimento",
        delta_color="inverse",
    )
with col_kpi3:
    st.metric(
        label="ROC-AUC Modello XGBoost", value="0.7600", delta="+0.08 vs RF"
    )

st.divider()

# --- MAPPA GEOGRAFICA DEI RITARDI ---
st.subheader("🗺️ Mappa Criticità Geografiche (Tasso di Ritardo per Stato)")

# Dati aggregati reali Olist per i 27 Stati Brasiliani
STATE_GEO_DATA = [
    {
        "state": "AL",
        "nome": "Alagoas",
        "lat": -9.57,
        "lon": -36.78,
        "delay_rate": 20.81,
    },
    {
        "state": "MA",
        "nome": "Maranhão",
        "lat": -5.42,
        "lon": -45.44,
        "delay_rate": 17.13,
    },
    {
        "state": "SE",
        "nome": "Sergipe",
        "lat": -10.57,
        "lon": -37.38,
        "delay_rate": 14.97,
    },
    {
        "state": "PI",
        "nome": "Piauí",
        "lat": -7.71,
        "lon": -42.72,
        "delay_rate": 13.80,
    },
    {
        "state": "CE",
        "nome": "Ceará",
        "lat": -5.20,
        "lon": -39.53,
        "delay_rate": 13.67,
    },
    {
        "state": "PB",
        "nome": "Paraíba",
        "lat": -7.24,
        "lon": -36.78,
        "delay_rate": 13.10,
    },
    {
        "state": "BA",
        "nome": "Bahia",
        "lat": -12.97,
        "lon": -38.51,
        "delay_rate": 12.65,
    },
    {
        "state": "RN",
        "nome": "Rio Grande do Norte",
        "lat": -5.22,
        "lon": -36.52,
        "delay_rate": 12.05,
    },
    {
        "state": "ES",
        "nome": "Espírito Santo",
        "lat": -19.18,
        "lon": -40.30,
        "delay_rate": 11.80,
    },
    {
        "state": "PE",
        "nome": "Pernambuco",
        "lat": -8.05,
        "lon": -34.88,
        "delay_rate": 11.50,
    },
    {
        "state": "RJ",
        "nome": "Rio de Janeiro",
        "lat": -22.90,
        "lon": -43.20,
        "delay_rate": 9.80,
    },
    {
        "state": "PA",
        "nome": "Pará",
        "lat": -1.99,
        "lon": -54.93,
        "delay_rate": 9.20,
    },
    {
        "state": "AM",
        "nome": "Amazonas",
        "lat": -3.07,
        "lon": -61.66,
        "delay_rate": 8.50,
    },
    {
        "state": "MT",
        "nome": "Mato Grosso",
        "lat": -12.64,
        "lon": -55.42,
        "delay_rate": 7.90,
    },
    {
        "state": "MS",
        "nome": "Mato Grosso do Sul",
        "lat": -20.77,
        "lon": -54.78,
        "delay_rate": 7.40,
    },
    {
        "state": "GO",
        "nome": "Goiás",
        "lat": -15.82,
        "lon": -49.83,
        "delay_rate": 7.10,
    },
    {
        "state": "DF",
        "nome": "Distrito Federal",
        "lat": -15.83,
        "lon": -47.86,
        "delay_rate": 6.80,
    },
    {
        "state": "MG",
        "nome": "Minas Gerais",
        "lat": -18.10,
        "lon": -44.38,
        "delay_rate": 5.40,
    },
    {
        "state": "RS",
        "nome": "Rio Grande do Sul",
        "lat": -30.03,
        "lon": -51.23,
        "delay_rate": 5.10,
    },
    {
        "state": "SC",
        "nome": "Santa Catarina",
        "lat": -27.24,
        "lon": -50.21,
        "delay_rate": 4.90,
    },
    {
        "state": "PR",
        "nome": "Paraná",
        "lat": -25.25,
        "lon": -52.02,
        "delay_rate": 4.60,
    },
    {
        "state": "SP",
        "nome": "São Paulo",
        "lat": -23.55,
        "lon": -46.63,
        "delay_rate": 4.43,
    },
    {
        "state": "RO",
        "nome": "Rondônia",
        "lat": -11.22,
        "lon": -62.80,
        "delay_rate": 4.20,
    },
    {
        "state": "TO",
        "nome": "Tocantins",
        "lat": -10.17,
        "lon": -48.33,
        "delay_rate": 4.10,
    },
    {
        "state": "AC",
        "nome": "Acre",
        "lat": -9.02,
        "lon": -70.81,
        "delay_rate": 3.80,
    },
    {
        "state": "AP",
        "nome": "Amapá",
        "lat": 0.90,
        "lon": -52.00,
        "delay_rate": 3.50,
    },
    {
        "state": "RR",
        "nome": "Roraima",
        "lat": 2.73,
        "lon": -61.30,
        "delay_rate": 3.20,
    },
]

df_geo = pd.DataFrame(STATE_GEO_DATA)

fig_map = px.scatter_geo(
    df_geo,
    lat="lat",
    lon="lon",
    size="delay_rate",
    color="delay_rate",
    hover_name="nome",
    hover_data={"state": True, "delay_rate": ":.2f%", "lat": False, "lon": False},
    color_continuous_scale="Reds",
    scope="south america",
    title="Tasso di Ritardo Consegne per Stato Destinazione (%)",
)

fig_map.update_geos(
    fitbounds="locations",
    visible=False,
    showcountries=True,
    countrycolor="#CCCCCC",
)

fig_map.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=450,
    coloraxis_colorbar=dict(title="% Ritardo"),
)

st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# --- FORM DI INPUT DELL'ORDINE ---
st.subheader("📋 Inserisci i dettagli dell'ordine")

col_left, col_right = st.columns(2)

# Lista degli Stati Brasiliani presenti nel dataset
STATES = [
    "SP",
    "RJ",
    "MG",
    "PR",
    "RS",
    "BA",
    "SC",
    "DF",
    "PE",
    "GO",
    "ES",
    "CE",
    "AL",
    "MA",
    "SE",
    "PA",
    "PB",
    "RN",
    "MT",
    "MS",
    "PI",
    "AM",
    "RO",
    "TO",
    "AC",
    "AP",
    "RR",
]

MONTHS = {
    1: "Gennaio",
    2: "Febbraio",
    3: "Marzo",
    4: "Aprile",
    5: "Maggio",
    6: "Giugno",
    7: "Luglio",
    8: "Agosto",
    9: "Settembre",
    10: "Ottobre",
    11: "Novembre",
    12: "Dicembre",
}

WEEKDAYS = {
    0: "Lunedì",
    1: "Martedì",
    2: "Mercoledì",
    3: "Giovedì",
    4: "Venerdì",
    5: "Sabato",
    6: "Domenica",
}

with col_left:
    totale_pagamento = st.number_input(
        "Importo Totale dell'Ordine (R$)",
        min_value=10.0,
        max_value=5000.0,
        value=150.0,
        step=10.0,
        help="Valore complessivo dei prodotti compreso il trasporto.",
    )

    stima_giorni_consegna = st.slider(
        "Giorni Stimati per la Consegna",
        min_value=1,
        max_value=60,
        value=15,
        help="Finestra temporale comunicata al cliente al checkout.",
    )

with col_right:
    customer_state = st.selectbox(
        "Stato di Destinazione del Cliente",
        options=STATES,
        index=0,  # Default SP
        help="Seleziona lo Stato brasiliano del destinatario.",
    )

    col_m, col_w = st.columns(2)
    with col_m:
        mese_acquisto = st.selectbox(
            "Mese dell'Acquisto",
            options=list(MONTHS.keys()),
            format_func=lambda x: MONTHS[x],
            index=10,  # Default Novembre (Black Friday)
        )
    with col_w:
        giorno_settimana = st.selectbox(
            "Giorno della Settimana",
            options=list(WEEKDAYS.keys()),
            format_func=lambda x: WEEKDAYS[x],
            index=0,
        )

# --- COSTRUZIONE INPUT DATASET PER PREDIZIONE ---
input_dict = {col: 0 for col in feature_names}

# Valorizzazione variabili numeriche
if "totale_pagamento" in input_dict:
    input_dict["totale_pagamento"] = totale_pagamento
if "stima_giorni_consegna" in input_dict:
    input_dict["stima_giorni_consegna"] = stima_giorni_consegna
if "mese_acquisto" in input_dict:
    input_dict["mese_acquisto"] = mese_acquisto
if "giorno_settimana" in input_dict:
    input_dict["giorno_settimana"] = giorno_settimana

# Valorizzazione One-Hot Encoding per lo Stato
state_col = f"customer_state_{customer_state}"
if state_col in input_dict:
    input_dict[state_col] = 1

input_df = pd.DataFrame([input_dict])

# --- CALCOLO PREDIZIONE ---
st.divider()

if st.button("Calcola Rischio Ritardo", type="primary", use_container_width=True):
    proba_delay = model.predict_proba(input_df)[0][1]
    is_delayed = proba_delay >= 0.50

    st.subheader("Risultato dell'Analisi Predittiva")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric(
            label="Probabilità di Ritardo Stimata",
            value=f"{proba_delay * 100:.1f}%",
        )

    with c2:
        if proba_delay >= 0.50:
            st.error(
                "**ALTO RISCHIO RITARDO**\n\nQuesto ordine presenta caratteristiche critiche (tratta o stagionalità a rischio). Si consiglia il pre-alerting logistico."
            )
        elif proba_delay >= 0.25:
            st.warning(
                "**RISCHIO MODERATO**\n\nL'ordine rientra in una fascia d'attenzione. Consigliato il monitoraggio dei tempi di spedizione."
            )
        else:
            st.success(
                "**BASSO RISCHIO RITARDO**\n\nL'ordine rientra nei parametri standard di consegna in orario."
            )