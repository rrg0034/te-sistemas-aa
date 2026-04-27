import random
from sqlalchemy.orm import Session
from .database import Base, engine, SessionLocal
from .models import User, Course, Interaction

AREAS = ["IA", "Big Data", "Python", "Cloud", "Ciberseguridad", "Bases de Datos"]
NIVELES = ["Inicial", "Intermedio", "Avanzado"]

CURSOS = [
    ("Python desde cero", "Python", "Inicial", 20),
    ("Python para análisis de datos", "Python", "Intermedio", 30),
    ("Machine Learning con Scikit-Learn", "IA", "Intermedio", 35),
    ("Deep Learning aplicado", "IA", "Avanzado", 45),
    ("Big Data con Spark", "Big Data", "Intermedio", 40),
    ("Arquitecturas Cloud", "Cloud", "Intermedio", 25),
    ("Seguridad en aplicaciones web", "Ciberseguridad", "Intermedio", 28),
    ("SQL y modelado de datos", "Bases de Datos", "Inicial", 22),
    ("MongoDB práctico", "Bases de Datos", "Intermedio", 18),
    ("MLOps y despliegue de modelos", "IA", "Avanzado", 32),
    ("Visualización de datos", "Big Data", "Inicial", 16),
    ("APIs con FastAPI", "Python", "Intermedio", 14),
    ("Introducción a Linux", "Cloud", "Inicial", 12),
    ("Análisis de logs de seguridad", "Ciberseguridad", "Avanzado", 26),
    ("Data Warehousing", "Big Data", "Avanzado", 34),
]


def create_database(reset: bool = False) -> None:
    if reset:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def seed_database() -> None:
    create_database(reset=True)
    db: Session = SessionLocal()

    random.seed(42)
    courses = []
    for titulo, categoria, nivel, duracion in CURSOS:
        course = Course(titulo=titulo, categoria=categoria, nivel=nivel, duracion_horas=duracion)
        db.add(course)
        courses.append(course)
    db.commit()

    for i in range(1, 81):
        area = random.choice(AREAS)
        nivel = random.choice(NIVELES)
        user = User(
            nombre=f"Usuario {i}",
            nivel=nivel,
            area_interes=area,
            frecuencia_acceso=round(random.uniform(1, 10), 2),
            duracion_media_sesion=round(random.uniform(10, 120), 2),
        )
        db.add(user)
        db.flush()

        cursos_preferidos = [c for c in courses if c.categoria == area or c.nivel == nivel]
        cursos_otros = [c for c in courses if c not in cursos_preferidos]
        elegidos = random.sample(cursos_preferidos, min(len(cursos_preferidos), random.randint(3, 6)))
        elegidos += random.sample(cursos_otros, random.randint(1, 3))

        for c in elegidos:
            match_bonus = 1.2 if c.categoria == area else 0
            progress = min(100, max(5, random.gauss(55 + match_bonus * 20, 25)))
            rating = min(5, max(1, random.gauss(3.2 + match_bonus, 0.9)))
            db.add(
                Interaction(
                    usuario_id=user.id,
                    curso_id=c.id,
                    progreso=round(progress, 2),
                    calificacion=round(rating, 2),
                    visualizaciones=random.randint(1, 12),
                    tiempo_total_min=round(random.uniform(15, c.duracion_horas * 60), 2),
                )
            )
    db.commit()
    db.close()


if __name__ == "__main__":
    seed_database()
    print("Base de datos creada en data/recomendador.db")
