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
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://phpmyadmin:askingme@localhost/absenaci2_absen'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
executor = ThreadPoolExecutor()

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
    # Pastikan known_encoding dalam bentuk list of encodings
    matches = face_recognition.compare_faces([known_encoding], face_encoding_to_check, tolerance=tolerance)
    return True in matches

def allowed_file(filename):
    """Memeriksa apakah ekstensi file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =============================================================================
# ENDPOINT STATIS
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

            # 1. Check if the provided user ID is already registered.
            if RegisteredFace.query.get(user_id):
                return {'message': f'ID pengguna {user_id} sudah terdaftar. Silakan gunakan ID lain.'}, 409

            # Read image from memory to avoid saving it unnecessarily
            photo_stream = photo.read()
            image_array = np.frombuffer(photo_stream, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                return {'message': 'Gagal membaca file gambar. Format mungkin tidak didukung.'}, 400

            face_encodings, _ = detect_face_encodings(image)

            if not face_encodings:
                return {'message': 'Tidak dapat menemukan wajah dalam foto'}, 400
            
            encoding_to_register = face_encodings[0]

            # 2. Check if the face itself is already registered under a different ID.
            all_users = RegisteredFace.query.filter(RegisteredFace.face_encoding.isnot(None)).all()
            for existing_user in all_users:
                try:
                    known_encoding = np.array(json.loads(existing_user.face_encoding))
                    is_match = compare_faces(known_encoding, encoding_to_register, tolerance=0.4) 
                    if is_match:
                        # Face already exists, so we abort the registration.
                        return {
                            'message': f'Wajah ini sudah terdaftar atas nama {existing_user.nama}. Registrasi dibatalkan.',
                            'user': {
                                'id': existing_user.id,
                                'nama': existing_user.nama,
                                'id_member': existing_user.id_member
                            }
                        }, 409 # HTTP 409 Conflict is the appropriate status code here.
                except Exception as e:
                    logging.warning(f"Error comparing with user ID {existing_user.id} during registration: {e}")
                    continue # Skip to the next user if one encoding is corrupted.

            # 3. If both ID and face are new, proceed with registration.
            filename = secure_filename(f"{user_id}_{name.replace(' ', '_')}_{photo.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file only after all checks have passed.
            photo.seek(0) # Rewind the file stream after it was read.
            photo.save(file_path)

            face_encoding_json = json.dumps(encoding_to_register.tolist())
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
                return {'message': 'Input tidak lengkap: user_id dan foto wajah diperlukan'}, 400

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
            
            # --- OPTIMISASI UTAMA ---
            if not user.face_encoding:
                return {'message': f'Encoding untuk user ID {user_id} tidak ditemukan. Mohon proses ulang data lama.'}, 400
            
            # Langsung ambil encoding dari database
            known_encoding = np.array(json.loads(user.face_encoding))

            # Baca gambar yang diupload dari memory, tidak perlu simpan ke disk
            photo_stream = photo.read()
            image_array = np.frombuffer(photo_stream, np.uint8)
            image_to_check = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Resize untuk proses lebih cepat
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
            

class CompareDirectAPI(Resource):
    def post(self):
        try:
            if 'photo' not in request.files:
                return {'message': 'Foto wajah diperlukan'}, 400

            photo = request.files['photo']

            if photo.filename == '' or not allowed_file(photo.filename):
                return {'message': 'Nama atau format file tidak valid'}, 400

            # Baca foto langsung dari memory
            photo_stream = photo.read()
            image_array = np.frombuffer(photo_stream, np.uint8)
            image_to_check = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            small_image = cv2.resize(image_to_check, (0, 0), fx=0.5, fy=0.5)

            future = executor.submit(detect_face_encodings, small_image)
            face_encodings_to_check, _ = future.result()

            if not face_encodings_to_check:
                return {'message': 'Tidak ada wajah yang terdeteksi pada foto'}, 400

            # Ambil encoding pertama (anggap hanya satu wajah dalam gambar)
            encoding_to_check = face_encodings_to_check[0]

            # Bandingkan dengan semua data di database
            users = RegisteredFace.query.filter(RegisteredFace.face_encoding.isnot(None)).all()

            for user in users:
                try:
                    known_encoding = np.array(json.loads(user.face_encoding))
                    is_match = compare_faces(known_encoding, encoding_to_check)
                    if is_match:
                        return {
                            'result': True,
                            'message': 'Wajah cocok dengan data yang terdaftar',
                            'user': {
                                'id': user.id,
                                'nama': user.nama,
                                'id_member': user.id_member,
                                'url': user.url_face_img
                            }
                        }, 200
                except Exception as e:
                    logging.warning(f"Gagal memproses user ID {user.id}: {e}")

            return {'result': False, 'message': 'Tidak ditemukan wajah yang cocok'}, 200

        except Exception as e:
            logging.error(f"CompareDirect Error: {e}")
            return {'message': 'Terjadi kesalahan internal'}, 500

class FaceListAPI(Resource):
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
        # ... (Logika PUT Anda, pastikan untuk menghitung ulang encoding jika foto diubah)
        # Implementasi PUT di bawah ini sudah diperbarui
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
                    # Hapus file lama
                    if os.path.exists(user.file_path):
                        os.remove(user.file_path)

                    # Simpan file baru
                    filename = secure_filename(f"{user.id}_{user.nama.replace(' ', '_')}_{photo.filename}")
                    new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    photo.save(new_file_path)
                    
                    image = cv2.imread(new_file_path)
                    face_encodings, _ = detect_face_encodings(image)

                    if not face_encodings:
                        os.remove(new_file_path)
                        db.session.rollback()
                        return {'message': 'Tidak dapat menemukan wajah di foto baru'}, 400

                    # Perbarui path, url, dan encoding
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
# =============================================================================
api.add_resource(RegisterAPI, '/api/register')
api.add_resource(CompareAPI, '/api/compare')
api.add_resource(CompareDirectAPI, '/api/compare_direct')
api.add_resource(FaceListAPI, '/api/faces')
api.add_resource(FaceAPI, '/api/faces/<int:face_id>')

# =============================================================================
# PERINTAH CLI UNTUK MEMPROSES DATA LAMA
# =============================================================================
@app.cli.command("process-existing-faces")
def process_existing_faces():
    """
    Membuat face encoding untuk semua data wajah yang sudah ada di database 
    namun belum memiliki encoding. Jalankan sekali setelah deploy kode baru.
    Cara menjalankan: flask process-existing-faces
    """
    with app.app_context():
        # Cari user yang encodingnya NULL atau string kosong
        users_to_process = RegisteredFace.query.filter(
            or_(RegisteredFace.face_encoding.is_(None), RegisteredFace.face_encoding == '')
        ).all()

        if not users_to_process:
            print("✅ Semua wajah yang terdaftar sudah memiliki encoding.")
            return

        print(f"Ditemukan {len(users_to_process)} wajah untuk diproses...")
        success_count, fail_count = 0, 0

        for user in users_to_process:
            print(f"Memproses user: {user.nama} (ID: {user.id})...")
            if not os.path.exists(user.file_path):
                print(f"  └── ⚠️ GAGAL: File gambar tidak ditemukan di '{user.file_path}'")
                fail_count += 1
                continue
            
            try:
                image = cv2.imread(user.file_path)
                if image is None:
                    raise Exception("Tidak bisa membaca file gambar")
                
                face_encodings, _ = detect_face_encodings(image)
                if not face_encodings:
                    raise Exception("Tidak ada wajah yang terdeteksi")
                
                user.face_encoding = json.dumps(face_encodings[0].tolist())
                db.session.add(user)
                print(f"  └── 👍 BERHASIL: Encoding untuk {user.nama} berhasil dibuat.")
                success_count += 1
            except Exception as e:
                print(f"  └── ❌ ERROR: Gagal memproses {user.nama}: {e}")
                fail_count += 1

        if success_count > 0:
            try:
                db.session.commit()
                print("\n✅ Semua perubahan berhasil disimpan ke database.")
            except Exception as e:
                db.session.rollback()
                print(f"\n❌ GAGAL menyimpan ke database: {e}")
        
        print("\n--- Ringkasan ---")
        print(f"Berhasil diproses: {success_count}")
        print(f"Gagal diproses: {fail_count}")

# =============================================================================
# MENJALANKAN APLIKASI
# =============================================================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5001)