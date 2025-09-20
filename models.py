from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class RegisteredFace(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=False)
    id_member = db.Column(db.Integer, nullable=True)
    nama = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    url_face_img = db.Column(db.String(255), nullable=True)
    face_encoding = db.Column(db.Text, nullable=False)

    __tablename__ = 'registered_faces'