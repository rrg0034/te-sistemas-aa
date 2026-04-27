from pydantic import BaseModel


class CourseOut(BaseModel):
    id: int
    titulo: str
    categoria: str
    nivel: str
    duracion_horas: float

    class Config:
        from_attributes = True


class UserOut(BaseModel):
    id: int
    nombre: str
    nivel: str
    area_interes: str
    frecuencia_acceso: float
    duracion_media_sesion: float

    class Config:
        from_attributes = True


class RecommendationOut(BaseModel):
    curso_id: int
    titulo: str
    categoria: str
    nivel: str
    puntuacion: float
    motivo: str


class TrainOut(BaseModel):
    usuarios: int
    cursos: int
    interacciones: int
    mejor_k: int
    silhouette_kmeans: float
    silhouette_dbscan: float | None
    precision_at_5: float
