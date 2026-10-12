# Laporan pembaruan UX dan kalibrasi buffer

Tanggal: 12 Oktober 2026.

## Perubahan utama

- Menyusun alur prediksi dalam tiga langkah yang jelas: unggah data, atur periode/buffer, dan baca hasil.
- Menampilkan daftar file terpilih, ukuran file, tombol hapus semua file, validasi format, pesan kesalahan yang lebih jelas, dan indikator proses.
- Membuat buffer otomatis sebagai pilihan utama; kolom buffer manual dijelaskan dan tingkat cakupan otomatis dinonaktifkan saat pengguna mengisi buffer manual.
- Memisahkan kebutuhan roasted/green tanpa buffer dari kebutuhan yang sudah ditambah buffer.
- Memindahkan pembaruan model operasional ke Pengaturan Lanjutan agar tidak bersaing dengan tombol prediksi utama.
- Memindahkan laporan preprocessing dan tabel teknis ke bagian rincian yang dapat dibuka bila diperlukan.
- Menambahkan navigasi drawer pada layar kecil, CTA dari dashboard ke prediksi, status muat/error, dan penanganan error pada halaman perbandingan.
- Menambahkan output `horizon` dan `model_source` pada respons prediksi agar label hasil web lebih akurat.
- Mengubah buffer otomatis: kalibrasi berasal dari ramalan rekursif periode validasi, bukan periode test final. Kalibrasi membedakan horizon 7, 14, dan 30 hari.
- Untuk horizon 60/90 hari, buffer otomatis menampilkan peringatan bahwa angka memakai acuan maksimum 30 hari yang tersedia dan harus dianggap perkiraan sementara, bukan kalibrasi horizon langsung.

## Hasil pengujian aktual

- `python -m py_compile app.py`: lulus.
- Pemeriksaan sintaks JavaScript untuk semua template dengan `node --check`: lulus.
- Parsing template Jinja: lulus.
- `python -m unittest discover -s tests -v` dengan stub lokal Flask/Werkzeug karena dependensi tersebut tidak terpasang di lingkungan ini: **9 tes lulus**.
- Pipeline penelitian penuh berhasil dijalankan: konfigurasi terpilih `RF + kalender dan proksi kampanye`.
- Kalibrasi buffer validasi menghasilkan 729 baris, yaitu 243 baris untuk tiap horizon 7, 14, dan 30 hari. Ada 9 titik awal ramalan pada data validasi.
- Backtest rekursif periode uji tetap berjalan terpisah dan menghasilkan 648 baris.
- Pembacaan rate penyusutan tetap tersedia untuk 27 dari 27 varian sampel.

## Batasan yang masih berlaku

- Flask/Werkzeug tidak tersedia pada environment pengembangan saat pemeriksaan ini, sehingga interaksi browser dan endpoint tidak bisa diuji end-to-end di sini. Setelah menginstal `requirements.txt`, jalankan aplikasi lokal untuk uji browser.
- Kalibrasi buffer dari 9 titik awal adalah sampel terbatas. Tingkat yang dipilih (80/90/95/99%) adalah kuantil kesalahan historis, bukan jaminan probabilistik bahwa stok selalu cukup.
- Horizon 60/90 hari belum memiliki kalibrasi rekursif khusus pada dataset validasi saat ini; antarmuka memberi tahu pengguna bahwa acuan 30 hari dipakai sebagai proksi.
- Fitur tanggal kembar dan gajian tetap berupa proksi kalender sampai data promosi aktual perusahaan tersedia.
- Beberapa pemetaan nama SKU ke tabel penyusutan masih membutuhkan konfirmasi perusahaan.
