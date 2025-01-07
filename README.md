# API ANPR Sederhana Menggunakan File Gambar

API ini digunakan untuk melakukan deteksi dan analisis plat nomor kendaraan secara otomatis (Automatic Number Plate Recognition - ANPR) menggunakan model TrOCR untuk OCR (Optical Character Recognition) dan YOLOv8 untuk deteksi objek. API ini menerima input berupa file gambar yang berisi plat nomor kendaraan, lalu mendeteksi teks pada plat dan melakukan analisis lebih lanjut berdasarkan plat nomor tersebut.

## Endpoint: `/prediction`

### Method: `POST`

#### Request:
- **Key**: `image` (file)
  - Gambar yang berisi plat nomor kendaraan yang akan dianalisis.
### Model TrOCR

- **V1 **: https://huggingface.co/Sans1807/APNR-Braincore-V2
- **V2 **: https://huggingface.co/Sans1807/APNR-Braincore-V1

### Model YoloV8 Object Detection

- **https://github.com/patriciasky17/apnr-braincore/tree/master/runs/detect/train9/weights**


### Response:
API akan mengembalikan hasil deteksi teks dari plat nomor pada gambar serta analisis dari nomor plat yang telah dideteksi.

### Contoh Response:
```json
{
  "nomor_plat": "B1234XYZ",
  "daerah": "Jakarta",
  "wilayah": "Jawa Barat",
  "jenis_kendaraan": "Kendaraan Penumpang",
  "image_path": "/path/to/saved/image.jpg"
}
