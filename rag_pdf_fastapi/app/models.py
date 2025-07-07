from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, func
from app.database import Base

class ChunkMetadata(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String(255), index=True)
    text = Column(Text)
    filename = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now())

