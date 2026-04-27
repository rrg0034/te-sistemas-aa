import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Recomendador de cursos", layout="wide")
st.title("Sistema de recomendación de cursos")
st.write("Clustering no supervisado + filtrado colaborativo basado en usuarios similares.")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Generar datos de prueba"):
        r = requests.post(f"{API_URL}/generar-datos", timeout=30)
        st.success(r.json()["mensaje"])
with col_b:
    if st.button("Entrenar y evaluar modelo"):
        r = requests.post(f"{API_URL}/entrenamiento", timeout=30)
        st.session_state["metrics"] = r.json()

if "metrics" in st.session_state:
    m = st.session_state["metrics"]
    st.subheader("Métricas de evaluación")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usuarios", m["usuarios"])
    c2.metric("Mejor K", m["mejor_k"])
    c3.metric("Silhouette KMeans", m["silhouette_kmeans"])
    c4.metric("Precisión@5", m["precision_at_5"])
    st.caption(f"Silhouette DBSCAN: {m['silhouette_dbscan']}")

try:
    users = requests.get(f"{API_URL}/usuarios", timeout=30).json()
    courses = requests.get(f"{API_URL}/cursos", timeout=30).json()
except Exception:
    st.warning("Arranca primero la API con: uvicorn app.main:app --reload")
    st.stop()

if not users:
    st.info("Pulsa 'Generar datos de prueba' para crear usuarios, cursos e interacciones.")
    st.stop()

st.subheader("Usuarios disponibles")
df_users = pd.DataFrame(users)
st.dataframe(df_users, use_container_width=True)

selected_user = st.selectbox(
    "Selecciona un usuario",
    options=df_users["id"].tolist(),
    format_func=lambda x: f"Usuario {x} - {df_users.loc[df_users['id'] == x, 'area_interes'].iloc[0]}",
)

top_k = st.slider("Número de recomendaciones", 1, 10, 5)

if st.button("Obtener recomendaciones"):
    recs = requests.get(f"{API_URL}/recomendar/{selected_user}?top_k={top_k}", timeout=30).json()
    st.subheader("Cursos recomendados")
    if recs:
        st.dataframe(pd.DataFrame(recs), use_container_width=True)
    else:
        st.warning("No se han podido generar recomendaciones para este usuario.")

st.subheader("Catálogo de cursos")
st.dataframe(pd.DataFrame(courses), use_container_width=True)
