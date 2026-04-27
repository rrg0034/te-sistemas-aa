from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .models import User, Course, Interaction


@dataclass
class ResultadoEntrenamiento:
    mejor_k: int
    silhouette_kmeans: float
    silhouette_dbscan: float | None
    precision_en_5: float
    etiquetas_cluster: np.ndarray


def cargar_datos(db: Session):
    usuarios = pd.read_sql(db.query(User).statement, db.bind)
    cursos = pd.read_sql(db.query(Course).statement, db.bind)
    interacciones = pd.read_sql(db.query(Interaction).statement, db.bind)

    return usuarios, cursos, interacciones


def construir_caracteristicas_usuarios(
    usuarios: pd.DataFrame,
    interacciones: pd.DataFrame) -> pd.DataFrame:
    datos_agrupados = interacciones.groupby("usuario_id").agg(
        progreso_medio=("progreso", "mean"),
        calificacion_media=("calificacion", "mean"),
        visualizaciones_total=("visualizaciones", "sum"),
        tiempo_total=("tiempo_total_min", "sum"),
        cursos_vistos=("curso_id", "nunique"),
    ).reset_index()

    datos = usuarios.merge(
        datos_agrupados,
        left_on="id",
        right_on="usuario_id",
        how="left"
    ).fillna(0)

    datos = pd.get_dummies(
        datos,
        columns=["nivel", "area_interes"],
        drop_first=False
    )

    return datos.drop(columns=["nombre", "usuario_id"], errors="ignore")


def optimizar_kmeans(caracteristicas: pd.DataFrame):
    datos_modelo = caracteristicas.drop(columns=["id"], errors="ignore")
    datos_escalados = StandardScaler().fit_transform(datos_modelo)

    mejor_k = 2
    mejor_silhouette = -1
    mejores_etiquetas = None

    for k in range(2, 8):
        modelo = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42
        )

        etiquetas = modelo.fit_predict(datos_escalados)
        silhouette = silhouette_score(datos_escalados, etiquetas)

        if silhouette > mejor_silhouette:
            mejor_k = k
            mejor_silhouette = silhouette
            mejores_etiquetas = etiquetas

    return mejor_k, float(mejor_silhouette), mejores_etiquetas, datos_escalados


def evaluar_dbscan(datos_escalados):
    mejor_silhouette = None

    for eps in [1.5, 2.0, 2.5, 3.0, 3.5]:
        etiquetas = DBSCAN(eps=eps, min_samples=4).fit_predict(datos_escalados)
        etiquetas_validas = set(etiquetas)
        if len(etiquetas_validas - {-1}) >= 2:
            silhouette = silhouette_score(datos_escalados, etiquetas)
            if mejor_silhouette is None or silhouette > mejor_silhouette:
                mejor_silhouette = float(silhouette)

    return mejor_silhouette


def crear_matriz_usuario_curso(interacciones: pd.DataFrame) -> pd.DataFrame:
    datos = interacciones.copy()

    datos["puntuacion"] = (datos["calificacion"] * 0.50 + (datos["progreso"] / 20) * 0.35 + np.log1p(datos["visualizaciones"]) * 0.15)
    matriz = datos.pivot_table(index="usuario_id", columns="curso_id", values="puntuacion", fill_value=0)

    return matriz


def calcular_precision_en_k(interacciones: pd.DataFrame, k: int = 5) -> float:
    matriz = crear_matriz_usuario_curso(interacciones)

    if matriz.empty or len(matriz) < 3:
        return 0.0

    generador = np.random.default_rng(42)
    aciertos = []

    for usuario_id in matriz.index:
        cursos_positivos = interacciones[
            (interacciones.usuario_id == usuario_id)
            & (interacciones.calificacion >= 4)
        ]

        if cursos_positivos.empty:
            continue

        curso_oculto = int(generador.choice(cursos_positivos.curso_id.values))

        matriz_entrenamiento = matriz.copy()

        if curso_oculto in matriz_entrenamiento.columns:
            matriz_entrenamiento.loc[usuario_id, curso_oculto] = 0

        similitudes = cosine_similarity(matriz_entrenamiento)

        posicion_usuario = matriz.index.get_loc(usuario_id)

        serie_similitudes = pd.Series(
            similitudes[posicion_usuario],
            index=matriz.index
        ).drop(usuario_id)

        usuarios_similares = serie_similitudes.sort_values(
            ascending=False
        ).head(10).index

        puntuaciones = matriz_entrenamiento.loc[usuarios_similares].mean(axis=0)

        cursos_ya_vistos = matriz_entrenamiento.loc[usuario_id][
            matriz_entrenamiento.loc[usuario_id] > 0
        ].index

        puntuaciones = puntuaciones.drop(
            index=cursos_ya_vistos,
            errors="ignore"
        )

        cursos_recomendados = puntuaciones.sort_values(
            ascending=False
        ).head(k).index.astype(int).tolist()

        aciertos.append(1 if curso_oculto in cursos_recomendados else 0)

    return float(np.mean(aciertos)) if aciertos else 0.0


def entrenar_modelo(db: Session) -> ResultadoEntrenamiento:
    usuarios, _, interacciones = cargar_datos(db)

    caracteristicas = construir_caracteristicas_usuarios(
        usuarios,
        interacciones
    )

    mejor_k, silhouette_kmeans, etiquetas, datos_escalados = optimizar_kmeans(
        caracteristicas
    )

    silhouette_dbscan = evaluar_dbscan(datos_escalados)

    precision_en_5 = calcular_precision_en_k(
        interacciones,
        k=5
    )

    return ResultadoEntrenamiento(
        mejor_k=mejor_k,
        silhouette_kmeans=silhouette_kmeans,
        silhouette_dbscan=silhouette_dbscan,
        precision_en_5=precision_en_5,
        etiquetas_cluster=etiquetas
    )


def recomendar_cursos(db: Session, usuario_id: int, top_k: int = 5):
    usuarios, cursos, interacciones = cargar_datos(db)

    if usuario_id not in usuarios.id.values:
        return []

    matriz = crear_matriz_usuario_curso(interacciones)

    if usuario_id not in matriz.index:
        return []

    similitudes = cosine_similarity(matriz)

    posicion_usuario = matriz.index.get_loc(usuario_id)

    serie_similitudes = pd.Series(
        similitudes[posicion_usuario],
        index=matriz.index
    ).drop(usuario_id)

    usuarios_similares = serie_similitudes.sort_values(
        ascending=False
    ).head(10).index

    puntuaciones = matriz.loc[usuarios_similares].mean(axis=0)

    cursos_ya_vistos = matriz.loc[usuario_id][
        matriz.loc[usuario_id] > 0
    ].index

    puntuaciones = puntuaciones.drop(
        index=cursos_ya_vistos,
        errors="ignore"
    )

    puntuaciones = puntuaciones.sort_values(
        ascending=False
    ).head(top_k)

    area_usuario = usuarios.loc[
        usuarios.id == usuario_id,
        "area_interes"
    ].iloc[0]

    recomendaciones = []

    for curso_id, puntuacion in puntuaciones.items():
        curso = cursos.loc[cursos.id == int(curso_id)].iloc[0]

        motivo = "Usuarios con un comportamiento parecido también valoraron bien este curso."

        if curso.categoria == area_usuario:
            motivo += " Además, el curso coincide con el área de interés del usuario."

        recomendaciones.append({
            "curso_id": int(curso.id),
            "titulo": curso.titulo,
            "categoria": curso.categoria,
            "nivel": curso.nivel,
            "puntuacion": round(float(puntuacion), 3),
            "motivo": motivo,
        })

    return recomendaciones