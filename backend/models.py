from sqlalchemy import Column, Integer, String, Float
from database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String)
    status = Column(String)  # pending / accepted / rejected
    result = Column(String)  # fake / real
    confidence = Column(Float)