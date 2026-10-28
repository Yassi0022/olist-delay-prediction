import joblib
import numpy as np
import pandas as pd
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