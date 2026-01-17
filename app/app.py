import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import base64
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_single, predict_batch

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Previsão de Sorvetes")
st.title("Previsão de Sorvetes por Temperatura")

st.header("Previsão única")
temp = st.number_input("Temperatura (°C)", value=25.0, step=0.1)

if st.button("Prever"):
    pred = predict_single({"temperatura": temp})
    st.write("Previsão:", round(pred, 2))

st.header("Previsão em lote")
csv_file = st.file_uploader("CSV com coluna temperatura", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.write(df.head())
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = Path(tmp.name)
        out = predict_batch(tmp_path)
        out_csv = out.to_csv(index=False).encode()
        b64 = base64.b64encode(out_csv).decode()
        link = f'<a href="data:file/csv;base64,{b64}" download="predicoes.csv">Baixar</a>'
        st.markdown(link, unsafe_allow_html=True)
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
