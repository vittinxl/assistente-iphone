"""
App Streamlit só para ML Tradicional (FLAML + UCI 350).
Nao precisa de Docker nem Flowise.

Como abrir: duplo-clique em INICIAR-ML.bat
"""

import streamlit as st

from ml_traditional.ui import render_ml_tab

st.set_page_config(
    page_title="ML Tradicional — Cartão de Crédito (UCI 350)",
    layout="wide",
)

st.title("ML Tradicional")
st.caption("FLAML AutoML · Dataset UCI 350 — Default of Credit Card Clients (Taiwan)")

render_ml_tab()
