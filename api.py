from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import os
import mysql.connector
from datetime import datetime
from http import HTTPStatus
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Konfigurasi Folder Upload dan Model
UPLOAD_FOLDER = 'log/gambar'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_OBJECT_DETECTION'] = './model/detect_plat.pt'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Inisialisasi Model Deteksi dan OCR
model_detect = YOLO(app.config['MODEL_OBJECT_DETECTION'])
detect_names = model_detect.names

processor_ocr = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
model_ocr = VisionEncoderDecoderModel.from_pretrained("Sans1807/APNR-Braincore-V2")

# Fungsi untuk Memeriksa Ekstensi File
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fungsi Koneksi ke Database
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="analytic"
    )

# Fungsi untuk Memperbarui Jumlah di Tabel Daerah
def update_daerah(nama_daerah):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT jumlah FROM daerah WHERE nama_daerah=%s", (nama_daerah,))
    result = cursor.fetchone()
    if result:
        cursor.execute("UPDATE daerah SET jumlah = jumlah + 1 WHERE nama_daerah=%s", (nama_daerah,))
    else:
        cursor.execute("INSERT INTO daerah (nama_daerah, jumlah) VALUES (%s, 1)", (nama_daerah,))
    conn.commit()
    cursor.close()
    conn.close()

# Fungsi untuk Memperbarui Jumlah di Tabel Wilayah
def update_wilayah(nama_wilayah):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT jumlah FROM wilayah WHERE nama_wilayah=%s", (nama_wilayah,))
    result = cursor.fetchone()
    if result:
        cursor.execute("UPDATE wilayah SET jumlah = jumlah + 1 WHERE nama_wilayah=%s", (nama_wilayah,))
    else:
        cursor.execute("INSERT INTO wilayah (nama_wilayah, jumlah) VALUES (%s, 1)", (nama_wilayah,))
    conn.commit()
    cursor.close()
    conn.close()

# Fungsi untuk Memasukkan Jenis Kendaraan ke dalam Tabel Kendaraan
def insert_jenis_kendaraan(nomor_plat, jenis_kendaraan):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT jumlah FROM jenis_kendaraan_counter WHERE jenis_kendaraan=%s", (jenis_kendaraan,))
    result = cursor.fetchone()
    if result:
        cursor.execute("UPDATE jenis_kendaraan_counter SET jumlah = jumlah + 1 WHERE jenis_kendaraan=%s", (jenis_kendaraan,))
    else:
        cursor.execute("INSERT INTO jenis_kendaraan_counter (jenis_kendaraan, jumlah) VALUES (%s, 1)", (jenis_kendaraan,))
    conn.commit()
    cursor.close()
    conn.close()

# Fungsi untuk Menyimpan Path Gambar ke dalam Database
def save_image_path(nomor_plat, image_path):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO uploaded_images (nomor_plat, image_path) VALUES (%s, %s)",
            (nomor_plat, image_path)
        )
        conn.commit()
    except mysql.connector.Error as err:
        app.logger.error(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()

# Fungsi untuk Memisahkan Nomor Plat menjadi Huruf Awal, Angka, dan Huruf Akhir
def pisahkan_nomor_plat(nomor_plat):
    match = re.match(r"([A-Z]+)(\d+)([A-Z]+)", nomor_plat)
    if match:
        return match.group(1), match.group(2), match.group(3)
    else:
        return None,None,None

# Peta Wilayah dan Daerah
sumatera_map = {
    "BL": "Aceh", "BB": "Sumatera Utara bagian barat", "BK": "Sumatera Utara bagian timur",
    "BA": "Sumatera Barat", "BM": "Riau", "BH": "Jambi", "BD": "Bengkulu", "BP": "Kepulauan Riau",
    "BG": "Sumatera Selatan", "BE": "Lampung"
}
banten_map = { "A": "Banten, Cilegon, Serang, Pandeglang, Lebak, Tangerang" }
jakarta_map = { "B": "Jakarta, Depok, Bekasi" }
jawa_barat_map = {
    "D": "Bandung", "E": "Cirebon, Majalengka, Indramayu, Kuningan", "F": "Bogor, Cianjur, Sukabumi",
    "T": "Purwakarta, Karawang, Subang", "Z": "Garut, Sumedang, Tasikmalaya, Pangandaran, Ciamis, Banjar"
}
jawa_tengah_map = {
    "G": "Pekalongan, Pemalang, Batang, Tegal, Brebes", "H": "Semarang, Kendal, Salatiga, Demak",
    "K": "Pati, Jepara, Kudus, Blora, Rembang", "R": "Banyumas, Purbalingga, Cilacap, Banjarnegara",
    "AA": "Magelang, Purworejo, Temanggung, Kebumen, Wonosobo", "AD": "Surakarta, Sukoharjo, Boyolali, Klaten"
}
jogja_map = { "AB": "Yogyakarta, Bantul, Gunung Kidul, Sleman, Kulon Progo" }
jawa_timur_map = {
    "L": "Surabaya", "M": "Madura", "N": "Malang, Pasuruan, Probolinggo, Lumajang",
    "P": "Bondowoso, Jember, Situbondo, Banyuwangi", "S": "Bojonegoro, Tuban, Mojokerto, Lamongan",
    "W": "Gresik, Sidoarjo", "AE": "Madiun, Ngawi, Ponorogo", "AG": "Kediri, Blitar, Tulungagung"
}
bali_nusa_map = {
    "DK": "Bali", "DR": "Pulau Lombok, Mataram", "EA": "Pulau Sumbawa", "DH": "Pulau Timor, Kupang",
    "EB": "Pulau Flores", "ED": "Pulau Sumba"
}
kalimantan_map = {
    "KB": "Singkawang, Pontianak", "DA": "Banjarmasin", "KH": "Palangkaraya, Kotawaringin, Barito",
    "KT": "Balikpapan, Samarinda, Bontang", "KU": "Kalimantan Utara"
}
sulawesi_map = {
    "DB": "Manado", "DL": "Sitaro, Talaud", "DM": "Gorontalo", "DN": "Palu, Poso",
    "DT": "Kendari, Konawe", "DD": "Makassar", "DC": "Majene"
}
maluku_papua_map = {
    "DE": "Maluku", "DG": "Ternate, Tidore", "PA": "Jayapura, Merauke", "PB": "Papua Barat"
}

# Fungsi untuk Mencari Daerah Berdasarkan Huruf Awal
def daerah_dari_huruf_awal(huruf_awal):
    maps = [
        (sumatera_map, "Sumatera"), (banten_map, "Banten"), (jakarta_map, "Jakarta"),
        (jawa_barat_map, "Jawa Barat"), (jawa_tengah_map, "Jawa Tengah"), (jogja_map, "Yogyakarta"),
        (jawa_timur_map, "Jawa Timur"), (bali_nusa_map, "Bali dan Nusa Tenggara"),
        (kalimantan_map, "Kalimantan"), (sulawesi_map, "Sulawesi"), (maluku_papua_map, "Maluku dan Papua")
    ]
    for daerah_map, wilayah in maps:
        if huruf_awal in daerah_map:
            nama_daerah = daerah_map[huruf_awal]
            update_daerah(nama_daerah)
            update_wilayah(wilayah)
            return nama_daerah, wilayah
    return "Daerah tidak diketahui", None

# Fungsi untuk Mengklasifikasikan Jenis Kendaraan
def klasifikasi_kendaraan(angka):
    angka = int(angka)
    if 1 <= angka <= 1999:
        return "Kendaraan Penumpang"
    elif 2000 <= angka <= 6999:
        return "Sepeda Motor"
    elif 7000 <= angka <= 7999:
        return "Bus"
    elif 8000 <= angka <= 9999:
        return "Kendaraan Berat atau Pengangkut"
    else:
        return "Tidak diketahui"

# Fungsi untuk Mencatat Log
def log_to_file(nomor_plat, nama_daerah, nama_wilayah, jenis_kendaraan):
    with open("log_plat.txt", "a") as log_file:
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{waktu} - Plat: {nomor_plat}, Daerah: {nama_daerah}, Wilayah: {nama_wilayah}, Jenis Kendaraan: {jenis_kendaraan}\n")

# Fungsi untuk Menyimpan Gambar yang Diunggah
def save_image(image):
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)
    return image_path

# Fungsi untuk Crop Plat Nomor dari Gambar
def crop(image_path):
    im0 = cv2.imread(image_path)
    if im0 is None:
        raise ValueError(f"Error reading image file {image_path}")

    results = model_detect.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=detect_names)

    if boxes:
        for box, cls in zip(boxes, clss):
            class_name = detect_names[int(cls)]
            annotator.box_label(box, color=colors(int(cls), True), label=class_name)
            crop_obj = im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
            return crop_obj
    else:
        raise ValueError("Tidak ditemukan plat nomor dalam gambar.")
    
# Fungsi untuk Melakukan OCR pada Gambar Plat Nomor
def ocr(image):
    pixel_values = processor_ocr(image, return_tensors='pt').pixel_values
    generated_ids = model_ocr.generate(pixel_values)
    generated_text = processor_ocr.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

@app.route('/api/gambar/<path:filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/prediction', methods=['POST'])
def prediction():
    if 'image' not in request.files:
        return jsonify({"error": "Gambar tidak ditemukan"}), HTTPStatus.BAD_REQUEST

    reqImage = request.files['image']
    if reqImage.filename == '':
        return jsonify({"error": "Nama file kosong"}), HTTPStatus.BAD_REQUEST

    if reqImage and allowed_file(reqImage.filename):
        try:
            image_path = save_image(reqImage)
            cropped_image = crop(image_path)
            plate_text = ocr(cropped_image)

            # Analisis Plat Nomor
            pisahan = pisahkan_nomor_plat(plate_text)
            if pisahan is None:
                return jsonify({"error": "Format nomor plat tidak valid"}), HTTPStatus.BAD_REQUEST
            huruf_awal, angka, huruf_akhir = pisahan
            nama_daerah, nama_wilayah = daerah_dari_huruf_awal(huruf_awal)
            jenis_kendaraan = klasifikasi_kendaraan(angka)
            insert_jenis_kendaraan(plate_text, jenis_kendaraan)
            save_image_path(plate_text, image_path)
            log_to_file(plate_text, nama_daerah, nama_wilayah, jenis_kendaraan)

            return jsonify({
                "data": {
                    "nomor_plat": plate_text,
                    "daerah": nama_daerah,
                    "wilayah": nama_wilayah,
                    "jenis_kendaraan": jenis_kendaraan,
                    "image_path": image_path
                }
            }), HTTPStatus.OK

        except Exception as e:
            app.logger.error(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR
    else:
        return jsonify({
            'status': {
                'code': HTTPStatus.BAD_REQUEST,
                'message': 'Format file tidak valid. Silakan unggah gambar dengan format JPG, PNG, atau JPEG.',
            }
        }), HTTPStatus.BAD_REQUEST

# Endpoint untuk Cek Status API
@app.route("/")
def index():
    return jsonify({
        "status" : {
            "code" : HTTPStatus.OK,
            "message" : "API Berhasil Terhubung",
        },
        "data" : None
    }), HTTPStatus.OK

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
