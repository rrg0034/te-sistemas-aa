from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from .database import Base, engine, get_db
from .models import User, Course, Interaction
from .schemas import CourseOut, UserOut, RecommendationOut, TrainOut
from .seed_data import seed_database
from .recommender import entrenar_modelo, recomendar_cursos

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sistema de recomendación de cursos",
    description="API con clustering no supervisado y filtrado colaborativo.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"mensaje": "API de recomendación activa", "docs": "/docs"}


@app.post("/generar-datos")
def seed():
    seed_database()
    return {"mensaje": "Datos de prueba generados correctamente"}


@app.get("/usuarios", response_model=list[UserOut])
def get_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.id).all()


@app.get("/cursos", response_model=list[CourseOut])
def get_courses(db: Session = Depends(get_db)):
    return db.query(Course).order_by(Course.id).all()


@app.post("/entrenamiento", response_model=TrainOut)
def train(db: Session = Depends(get_db)):
    result = entrenar_modelo(db)
    return {
        "usuarios": db.query(User).count(),
        "cursos": db.query(Course).count(),
        "interacciones": db.query(Interaction).count(),
        "mejor_k": result.mejor_k,
        "silhouette_kmeans": round(result.silhouette_kmeans, 4),
        "silhouette_dbscan": round(result.silhouette_dbscan, 4) if result.silhouette_dbscan is not None else None,
        "precision_at_5": round(result.precision_en_5, 4),
    }


@app.get("/recomendar/{user_id}", response_model=list[RecommendationOut])
def recommend(user_id: int, top_k: int = 5, db: Session = Depends(get_db)):
    if db.query(User).filter(User.id == user_id).first() is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return recomendar_cursos(db, user_id=user_id, top_k=top_k)
