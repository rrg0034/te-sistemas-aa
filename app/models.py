from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.sql import func
from .database import Base


class User(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, nullable=False)
    nivel = Column(String, nullable=False)
    area_interes = Column(String, nullable=False)
    frecuencia_acceso = Column(Float, nullable=False)
    duracion_media_sesion = Column(Float, nullable=False)


class Course(Base):
    __tablename__ = "cursos"

    id = Column(Integer, primary_key=True, index=True)
    titulo = Column(String, nullable=False)
    categoria = Column(String, nullable=False)
    nivel = Column(String, nullable=False)
    duracion_horas = Column(Float, nullable=False)


class Interaction(Base):
    __tablename__ = "interacciones"

    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    curso_id = Column(Integer, ForeignKey("cursos.id"), nullable=False)
    progreso = Column(Float, nullable=False)
    calificacion = Column(Float, nullable=False)
    visualizaciones = Column(Integer, nullable=False)
    tiempo_total_min = Column(Float, nullable=False)
    fecha = Column(DateTime, server_default=func.now())
