import os
import shutil
from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session

# 🔥 DEBUG STARTUP INFO
print("🚀 STARTING APP...")
print("📁 Current directory:", os.getcwd())
print("📦 Files in directory:", os.listdir())
print("🌐 DATABASE_URL:", os.getenv("DATABASE_URL"))

# 🔥 SAFE IMPORTS
try:
    from database import SessionLocal, engine
    print("✅ Database module loaded")
except Exception as e:
    print("❌ Database import error:", str(e))

try:
    import models
    print("✅ Models loaded")
except Exception as e:
    print("❌ Models import error:", str(e))

try:
    from ml import predict_image
    print("✅ ML module loaded")
except Exception as e:
    print("❌ ML import error:", str(e))

# 🔥 CREATE TABLES SAFELY
try:
    models.Base.metadata.create_all(bind=engine)
    print("✅ Tables created")
except Exception as e:
    print("❌ DB table creation error:", str(e))


app = FastAPI()

# ✅ FIX UPLOAD PATH (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
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
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

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

    print("📂 Verifying file:", doc.file_path)

    try:
        result = predict_image(doc.file_path)
    except Exception as e:
        print("❌ ML runtime error:", str(e))
        return {"error": str(e)}

    doc.result = result.get("result")
    doc.confidence = result.get("confidence")
    doc.status = "accepted" if result.get("result") == "real" else "rejected"

    db.commit()

    return result


# Get all documents (admin)
@app.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    return db.query(models.Document).all()