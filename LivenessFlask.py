# app.py

import os
import cv2
import face_recognition
import logging
import json
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, render_template, url_for
from flask_restful import Api, Resource
from models import db, RegisteredFace
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_swagger_ui import get_swaggerui_blueprint
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import or_

# <<< BARU DIMULAI: Import untuk Liveness Detection >>>
import dlib
from scipy.spatial import distance as dist
# <<< BARU SELESAI: Import untuk Liveness Detection >>>

# =============================================================================
# KONFIGURASI APLIKASI
# =============================================================================
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
api = Api(app)
CORS(app)

# Konfigurasi Folder
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Konfigurasi Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://apifr:askingme@localhost/db_apifr'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
executor = ThreadPoolExecutor()

# <<< BARU DIMULAI: Konfigurasi untuk Liveness Detection >>>
# Pastikan file ini ada di direktori root proyek Anda
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(SHAPE_PREDICTOR_PATH):
    logging.error(f"FATAL: File landmark predictor tidak ditemukan di '{SHAPE_PREDICTOR_PATH}'")
    # Anda bisa menghentikan aplikasi jika file tidak ada
    # exit() 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Threshold untuk mata dianggap terbuka. Sesuaikan jika perlu.
EYE_AR_THRESH = 0.22
# <<< BARU SELESAI: Konfigurasi untuk Liveness Detection >>>


# =============================================================================
# KONFIGURASI SWAGGER UI
# =============================================================================
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL, config={'app_name': "Face Recognition API V3"}
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

# =============================================================================
# FUNGSI BANTUAN
# =============================================================================
def detect_face_encodings(image):
    """Mendeteksi dan menghasilkan encoding dari sebuah gambar."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_encodings, face_locations

def compare_faces(known_encoding, face_encoding_to_check, tolerance=0.4):
    """Membandingkan satu encoding yang diketahui dengan satu encoding yang akan diperiksa."""
    matches = face_recognition.compare_faces([known_encoding], face_encoding_to_check, tolerance=tolerance)
    return True in matches

def allowed_file(filename):
    """Memeriksa apakah ekstensi file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# <<< BARU DIMULAI: Fungsi untuk Liveness Detection >>>
def calculate_ear(eye):
    """Menghitung Eye Aspect Ratio (EAR) untuk satu mata."""
    # Jarak vertikal
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Jarak horizontal
    C = dist.euclidean(eye[0], eye[3])
    # Hitung EAR
    ear = (A + B) / (2.0 * C)
    return ear

def check_liveness(image):
    """
    Memeriksa liveness dengan memastikan mata terbuka (EAR > threshold).
    Mengembalikan True jika mata terbuka, False jika tidak.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        return False, "Wajah tidak terdeteksi oleh dlib"

    # Ambil wajah pertama yang terdeteksi
    rect = rects[0]
    shape = predictor(gray, rect)
    
    # Konversi koordinat landmark ke NumPy array
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # Ekstrak koordinat mata kiri dan kanan
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)
    leftEye = coords[lStart:lEnd]
    rightEye = coords[rStart:rEnd]

    # Hitung EAR untuk kedua mata
    leftEAR = calculate_ear(leftEye)
    rightEAR = calculate_ear(rightEye)

    # Rata-ratakan EAR
    ear = (leftEAR + rightEAR) / 2.0
    
    logging.info(f"Calculated EAR: {ear:.4f}")

    # Jika EAR di bawah threshold, anggap mata tertutup atau bukan wajah hidup
    if ear < EYE_AR_THRESH:
        return False, f"Liveness check gagal (EAR: {ear:.2f} < {EYE_AR_THRESH})"
    
    return True, "Liveness check berhasil"
# <<< BARU SELESAI: Fungsi untuk Liveness Detection >>>


# =============================================================================
# ENDPOINT STATIS
# ... (Tidak ada perubahan di sini)
# =============================================================================
@app.route('/')
def index():
    try:
        users = RegisteredFace.query.all()
        return render_template('index.html', users=users)
    except Exception as e:
        return str(e), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# =============================================================================
# KELAS RESOURCE UNTUK API RESTful
# =============================================================================
class RegisterAPI(Resource):
    # ... (Tidak ada perubahan di sini)
    def post(self):
        try:
            if 'photo' not in request.files or 'name' not in request.form or 'id' not in request.form:
                return {'message': 'Input tidak lengkap: photo, name, id diperlukan'}, 400

            photo = request.files['photo']
            name = request.form['name']
            user_id_str = request.form['id']
            member_id_raw = request.form.get('member_id')
            member_id = int(member_id_raw) if member_id_raw and member_id_raw.strip() != '' else None

            if photo.filename == '' or not allowed_file(photo.filename):
                return {'message': 'Nama atau format file tidak valid'}, 400

            try:
                user_id = int(user_id_str)
            except ValueError:
                return {'message': 'ID harus berupa angka (integer)'}, 400

            if RegisteredFace.query.get(user_id):
                return {'message': f'ID {user_id} sudah terdaftar'}, 409

            filename = secure_filename(f"{user_id}_{name.replace(' ', '_')}_{photo.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(file_path)

            image = cv2.imread(file_path)
            face_encodings, _ = detect_face_encodings(image)

            if not face_encodings:
                os.remove(file_path)
                return {'message': 'Tidak dapat menemukan wajah dalam foto'}, 400
            
            face_encoding_json = json.dumps(face_encodings[0].tolist())

            base_url = os.environ.get('APP_BASE_URL', request.host_url.rstrip('/'))
            url_face_img = f"{base_url}{url_for('uploaded_file', filename=filename)}"

            new_face = RegisteredFace(
                id=user_id,
                nama=name,
                id_member=member_id,
                file_path=file_path,
                url_face_img=url_face_img,
                face_encoding=face_encoding_json
            )
            db.session.add(new_face)
            db.session.commit()
            return {'message': 'Foto berhasil diregistrasi', 'data': {'id': new_face.id, 'nama': new_face.nama, 'id_member': new_face.id_member, 'url': new_face.url_face_img}}, 201

        except Exception as e:
            db.session.rollback()
            logging.error(f"Register Error: {e}")
            return {'message': 'Terjadi kesalahan internal'}, 500

class CompareAPI(Resource):
    def post(self):
        try:
            if 'user_id' not in request.form or 'photo' not in request.files:
                return {'message': 'Input tidak lengkap: user_id dan photo diperlukan'}, 400

            user_id_str = request.form['user_id']
            photo = request.files['photo']

            if photo.filename == '' or not allowed_file(photo.filename):
                return {'message': 'Nama atau format file tidak valid'}, 400

            try:
                user_id = int(user_id_str)
            except ValueError:
                return {'message': 'ID harus berupa angka (integer)'}, 400

            user = RegisteredFace.query.get(user_id)
            if not user:
                return {'message': f'User dengan ID {user_id} tidak ditemukan!'}, 404
            
            if not user.face_encoding:
                return {'message': f'Encoding untuk user ID {user_id} tidak ditemukan. Mohon proses ulang data lama.'}, 400
            
            known_encoding = np.array(json.loads(user.face_encoding))

            photo_stream = photo.read()
            image_array = np.frombuffer(photo_stream, np.uint8)
            image_to_check = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # <<< BARU DIMULAI: Integrasi Liveness Check >>>
            is_live, liveness_message = check_liveness(image_to_check)
            if not is_live:
                logging.warning(f"Liveness check failed for user {user_id}: {liveness_message}")
                return {'message': 'Pengecekan keaslian wajah gagal. Pastikan wajah terlihat jelas dan mata terbuka.'}, 400
            # <<< BARU SELESAI: Integrasi Liveness Check >>>

            small_image = cv2.resize(image_to_check, (0, 0), fx=0.5, fy=0.5)

            future = executor.submit(detect_face_encodings, small_image)
            face_encodings_to_check, _ = future.result()

            if face_encodings_to_check:
                is_recognized = compare_faces(known_encoding, face_encodings_to_check[0])
                if is_recognized:
                    return {'result': True, 'message': 'Wajah dikenali', 'user': {'id': user.id, 'nama': user.nama, 'id_member': user.id_member}}, 200
                else:
                    return {'result': False, 'message': 'Wajah tidak cocok'}, 200
            else:
                return {'message': 'Wajah tidak terdeteksi pada foto yang diunggah'}, 400

        except Exception as e:
            logging.error(f"Compare Error: {e}")
            return {'message': 'Terjadi kesalahan internal'}, 500

class FaceListAPI(Resource):
    # ... (Tidak ada perubahan di sini)
    def get(self):
        id_member = request.args.get('id_member')
        try:
            if id_member:
                users = RegisteredFace.query.filter_by(id_member=int(id_member)).all()
            else:
                users = RegisteredFace.query.all()
            
            return jsonify({'registered_faces': [
                {'id': user.id, 'nama': user.nama, 'id_member': user.id_member, 'url_face_img': user.url_face_img}
                for user in users
            ]})
        except ValueError:
            return {'message': 'id_member harus berupa angka'}, 400
        except Exception as e:
            logging.error(f"FaceList Error: {e}")
            return {'message': 'Terjadi kesalahan internal'}, 500

class FaceAPI(Resource):
    # ... (Tidak ada perubahan di sini)
    def get(self, face_id):
        user = RegisteredFace.query.get(face_id)
        if not user:
            return {'message': 'User tidak ditemukan'}, 404
        return jsonify({'id': user.id, 'nama': user.nama, 'id_member': user.id_member, 'url_face_img': user.url_face_img})

    def delete(self, face_id):
        user = RegisteredFace.query.get(face_id)
        if not user:
            return {'message': 'User tidak ditemukan'}, 404
        
        if os.path.exists(user.file_path):
            os.remove(user.file_path)
        
        db.session.delete(user)
        db.session.commit()
        return {'message': f'Wajah dengan ID {face_id} berhasil dihapus'}, 200

    def put(self, face_id):
        try:
            user = RegisteredFace.query.get(face_id)
            if not user:
                return {'message': 'User tidak ditemukan'}, 404

            if 'name' in request.form:
                user.nama = request.form['name']
            
            if 'member_id' in request.form:
                member_id_raw = request.form.get('member_id')
                user.id_member = int(member_id_raw) if member_id_raw and member_id_raw.strip() != '' else None

            if 'photo' in request.files:
                photo = request.files['photo']
                if photo.filename != '' and allowed_file(photo.filename):
                    if os.path.exists(user.file_path):
                        os.remove(user.file_path)

                    filename = secure_filename(f"{user.id}_{user.nama.replace(' ', '_')}_{photo.filename}")
                    new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    photo.save(new_file_path)
                    
                    image = cv2.imread(new_file_path)
                    face_encodings, _ = detect_face_encodings(image)

                    if not face_encodings:
                        os.remove(new_file_path)
                        db.session.rollback()
                        return {'message': 'Tidak dapat menemukan wajah di foto baru'}, 400

                    user.file_path = new_file_path
                    base_url = os.environ.get('APP_BASE_URL', request.host_url.rstrip('/'))
                    user.url_face_img = f"{base_url}{url_for('uploaded_file', filename=filename)}"
                    user.face_encoding = json.dumps(face_encodings[0].tolist())
            
            db.session.commit()
            return {'message': 'Data berhasil diupdate', 'data': {'id': user.id, 'nama': user.nama, 'id_member': user.id_member, 'url': user.url_face_img}}, 200

        except Exception as e:
            db.session.rollback()
            logging.error(f"Update Error: {e}")
            return {'message': 'Terjadi kesalahan internal'}, 500

# =============================================================================
# MAPPING ENDPOINT API
# ... (Tidak ada perubahan di sini)
# =============================================================================
api.add_resource(RegisterAPI, '/api/register')
api.add_resource(CompareAPI, '/api/compare')
api.add_resource(FaceListAPI, '/api/faces')
api.add_resource(FaceAPI, '/api/faces/<int:face_id>')

# =============================================================================
# PERINTAH CLI
# ... (Tidak ada perubahan di sini)
# =============================================================================
@app.cli.command("process-existing-faces")
def process_existing_faces():
    """
    Membuat face encoding untuk semua data wajah yang sudah ada di database 
    namun belum memiliki encoding. Jalankan sekali setelah deploy kode baru.
    Cara menjalankan: flask process-existing-faces
    """
    # ... (Isi fungsi ini tidak berubah)
    pass # Placeholder untuk keringkasan, isi fungsi ini sama seperti file asli Anda


# =============================================================================
# MENJALANKAN APLIKASI
# =============================================================================
if __name__ == '__main__':
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"ERROR: file '{SHAPE_PREDICTOR_PATH}' tidak ditemukan. Silakan unduh dan letakkan di direktori proyek.")
    else:
        with app.app_context():
            db.create_all()
        app.run(host='0.0.0.0', port=5001)