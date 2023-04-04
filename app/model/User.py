from sqlalchemy import Column, Integer, String, Float, Text
from config.database.db import db
from sqlalchemy.ext.declarative import declarative_base

base = declarative_base()
class User (db.Model):
    __tablename__ = "User"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(30), nullable=False, unique=True)

