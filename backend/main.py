from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
import shutil
import os

from database import SessionLocal, engine
import models
from ml import predict_image

# Create DB tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# DB connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def home():
    return {"message": "Secure Doc Verify API running 🚀"}


# Upload document
@app.post("/upload")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc = models.Document(
        file_path=file_path,
        status="pending"
    )

    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {"doc_id": doc.id}


# Verify document using ML
@app.post("/verify/{doc_id}")
def verify(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()

    if not doc:
        return {"error": "Document not found"}

    result = predict_image(doc.file_path)

    doc.result = result["result"]
    doc.confidence = result["confidence"]
    doc.status = "accepted" if result["result"] == "real" else "rejected"

    db.commit()

    return result


# Get all documents (admin)
@app.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    return db.query(models.Document).all()