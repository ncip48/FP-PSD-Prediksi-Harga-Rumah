# Deskripsi Proyek: Prediksi Menggunakan Regresi Linier Berganda

## Tujuan Proyek

Proyek ini bertujuan untuk menganalisis data dan membangun sebuah model prediksi menggunakan metode **Regresi Linier Berganda (Multiple Linear Regression)**. Metode statistik ini digunakan untuk mengetahui bagaimana beberapa faktor yang berbeda secara bersama-sama memengaruhi satu hasil akhir.

## Cara Kerja Singkat

Berbeda dengan regresi linier sederhana yang hanya melihat satu sebab-akibat, regresi linier berganda menggunakan dua atau lebih faktor penentu. Sistem akan mempelajari data historis yang ada untuk mencari pola, sehingga kita dapat mengukur seberapa besar pengaruh masing-masing faktor terhadap variabel target.

## Contoh Penerapan

Sebagai contoh, jika proyek ini bertujuan untuk memprediksi **Harga Rumah**, maka pembagian variabelnya adalah:

### Variabel Dependen (Target)
- Harga rumah

### Variabel Independen (Faktor Penentu)
- Luas tanah
- Luas bangunan

## Tahapan Utama Proyek

### 1. Pengumpulan Data
Menyiapkan kumpulan data yang berisi nilai dari variabel independen beserta nilai target yang akan diprediksi.

### 2. Analisis Data
Melakukan pemeriksaan dan pembersihan data agar siap diproses, seperti:
- Menghapus data kosong (*missing values*)
- Memperbaiki data yang tidak valid
- Menangani data duplikat jika diperlukan

### 3. Pemodelan
Melatih model **Regresi Linier Berganda** menggunakan data historis untuk menemukan hubungan matematis antara variabel independen dan variabel dependen.

### 4. Evaluasi
Menguji performa model menggunakan data uji untuk mengetahui tingkat akurasi prediksi terhadap data yang belum pernah dilihat sebelumnya.

## Hasil yang Diharapkan

Pada akhir proyek, diharapkan diperoleh:

- Model prediksi yang mampu memperkirakan nilai target dengan tingkat akurasi yang baik.
- Informasi mengenai pengaruh masing-masing variabel independen terhadap hasil prediksi.
- Insight mengenai faktor yang paling dominan dalam memengaruhi variabel target.
- Dasar pengambilan keputusan yang lebih objektif berdasarkan hasil analisis data.

## Struktur Direktori
- `dataset/` : Berisi file data CSV (contoh: rumah123_malang.csv)
- `output/` : Folder tempat visualisasi dari korelasi dan distribusi residual disimpan dalam bentuk PNG.
- `utils.py` : Modul helper yang mencakup fungsi untuk *data cleaning*, training model regresi linier, dan pengujian asumsi klasik.
- `main.py` : Script utama yang menjalankan keseluruhan *pipeline* regresi (dari ekstraksi fitur hingga testing).
- `hasil_analisis.txt` : Output teks yang digenerate oleh script utama berisi *R-squared*, *MAE*, *RMSE*, hasil multikolinearitas, heteroskedastisitas, dll.