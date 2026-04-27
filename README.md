# Sistema de recomendación de cursos
Proyecto para el trabajo de enfoque, Sistemas de Aprendizaje Automático.

## Tecnologías
- Python
- FastAPI
- SQLite
- SQLAlchemy
- Pandas
- Scikit-Learn
- Matplotlib
- Streamlit

## Funcionalidades
- Generación de datos de usuarios, cursos e interacciones.
- Segmentación de usuarios con K-Means.
- Comparación con DBSCAN.
- Optimización de parámetros mediante índice de Silhouette.
- Recomendación de cursos mediante filtrado colaborativo basado en similitud entre usuarios.
- Evaluación mediante Precisión@K.
- API REST con FastAPI.
- Interfaz visual con Streamlit.

## Instalación
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución
Desde la carpeta raíz del proyecto:

```bash
uvicorn app.main:app --reload
```

En otra terminal (Front):

```bash
streamlit run streamlit_app/app.py
```

## Uso
1. Abrir la API en http://127.0.0.1:8000/docs
2. Ejecutar `POST /generar-datos` para generar datos.
3. Ejecutar `POST /entrenamiento` para entrenar y evaluar.
4. Ejecutar `GET /recomendar/{user_id}` para obtener recomendaciones.
5. Abrir Streamlit para visualizar usuarios, métricas y recomendaciones.

## Estructura
```text
recomendador_cursos_ml/
├── app/
|   ├── screenshots/
|   |   └── imgs
|   ├── __init__.py
│   ├── database.py
│   ├── main.py
│   ├── models.py
│   ├── recommender.py
│   ├── schemas.py
│   └── seed_data.py
├── streamlit_app/
│   └── app.py
├── venv/
├── requirements.txt
└── README.md
```
