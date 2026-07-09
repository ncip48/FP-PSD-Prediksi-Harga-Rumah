# Project Regresi Harga Rumah Malang

## Struktur Direktori
- `dataset/` : Berisi file data CSV (contoh: rumah123_malang.csv)
- `output/` : Folder tempat visualisasi dari korelasi dan distribusi residual disimpan dalam bentuk PNG.
- `utils.py` : Modul helper yang mencakup fungsi untuk *data cleaning*, training model regresi linier, dan pengujian asumsi klasik.
- `main.py` : Script utama yang menjalankan keseluruhan *pipeline* regresi (dari ekstraksi fitur hingga testing).
- `hasil_analisis.txt` : Output teks yang digenerate oleh script utama berisi *R-squared*, *MAE*, *RMSE*, hasil multikolinearitas, heteroskedastisitas, dll.