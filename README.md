# Semekar — Prediksi Penjualan dan Kebutuhan Bahan Baku Kopi

Aplikasi Flask untuk memprediksi penjualan dalam gram per varian menggunakan Random Forest Regression, mengevaluasi model secara temporal, dan mengonversi ramalan menjadi kebutuhan roasted bean serta estimasi green bean.

## Menjalankan aplikasi

1. Gunakan Python 3.10+ (Python 3.11 atau 3.12 disarankan).
2. Buka terminal pada folder proyek.
3. Buat dan aktifkan virtual environment.
4. Pasang dependensi: `pip install -r requirements.txt`.
5. Salin `.env.example` menjadi `.env`, lalu ubah `APP_USER`, `APP_PASS`, dan `SECRET_KEY` sebelum dipakai di luar komputer lokal.
6. Jalankan `python app.py`, lalu buka `http://127.0.0.1:5000`.

Pada proses awal aplikasi membangun cache model penelitian dari `data/Dataset_Sampling_v2.xlsx`. Proses pertama dapat memerlukan waktu lebih lama. File cache dibuat otomatis dan tidak perlu dimasukkan ke arsip proyek.

## Alur prediksi kebutuhan bahan baku

1. Buka **Prediksi kebutuhan bahan baku** dan unggah file transaksi terbaru CSV/XLSX. Beberapa file dari berbagai kanal dapat dipilih sekaligus.
2. Pilih horizon prediksi 7–90 hari dan buffer kebutuhan jika diperlukan, kemudian tekan **Praproses & prediksi**.
3. Aplikasi menampilkan estimasi penjualan dalam gram per varian, total kebutuhan roasted bean, serta estimasi green bean yang dikonversi dari penyusutan pada `data/Stock.xlsx`, sheet `Penyusutan`.
4. Tidak perlu mengunggah file persediaan untuk menghitung kebutuhan bruto. Buffer otomatis dikalibrasi dari kesalahan ramalan rekursif pada periode validasi, bukan dari periode test akhir. Untuk horizon 60/90 hari, acuan maksimum 30 hari digunakan sebagai pendekatan sementara dan ditandai pada antarmuka.
5. Tombol **Validasi & perbarui model** merupakan proses terpisah: unggah riwayat transaksi yang cukup panjang, lalu sistem menguji model pada periode terbaru yang ditahan dari pelatihan. Model operasional hanya diaktifkan jika WAPE Random Forest tidak lebih buruk daripada baseline rata-rata 7 hari.

Pembaruan operasional disimpan di `operational_model.joblib`, terpisah dari evaluasi penelitian. Hasil penelitian (training sampai 30 Juni 2025, validasi Juli–September 2025, dan test Oktober–24 Desember 2025) tidak berubah hanya karena model operasional diperbarui.

## Konversi roasted bean ke green bean

Nama pada `Referensi_Varian.nama_di_tabel_penyusutan` digunakan untuk memetakan varian penjualan ke `Stock.xlsx` sheet `Penyusutan`. Untuk persentase penyusutan `s`, perhitungan adalah:

`green bean (g) = roasted bean (g) / (1 - s)`

Contoh: penyusutan 17% dan kebutuhan 10.000 g roasted menghasilkan estimasi sekitar 12.048 g green bean. Tabel menunjukkan gram/kg roasted, rate penyusutan, faktor konversi, estimasi gram/kg green, dan status kecocokan nama produk. Untuk nama/SKU yang belum dikonfirmasi, estimasi ditampilkan sebagai sementara.

**Catatan:** keluaran green bean adalah konversi matematis dari forecast roasted, bukan model machine learning kedua. Hasil prediksi roasted/green merupakan kebutuhan bruto, bukan pembelian neto setelah dikurangi stok gudang. Pengurangan stok, pesanan masuk, dan target stok akhir hanya diperlukan jika nanti perusahaan membutuhkan modul perencanaan pembelian neto yang terpisah.

Saldo pada `Stock.xlsx` sheet `Sheet8` tidak otomatis dipakai karena merupakan snapshot bertanggal 19 Januari 2026 dan tidak boleh diasumsikan sebagai saldo gudang terkini.

## Evaluasi penelitian

Model dipilih berdasarkan WAPE periode validasi. Periode test hanya untuk pelaporan. Baris tanggal tidak lengkap/imputasi dikeluarkan dari metrik utama. Eksperimen target clipping p99,5 tersedia sebagai pembanding; data aktual test tetap tidak dipangkas. Perbandingan metode dapat dilihat pada halaman **Perbandingan metode**.

## Pengujian

Setelah dependensi terpasang, jalankan dari folder proyek:

```bash
python -m unittest discover -s tests -v
```

Tes mencakup fitur lag tanpa target leakage, metrik target nol, perhitungan stok neto (fungsi terpisah), pengecualian tanggal imputasi, validasi minimum untuk pembaruan model, pembacaan rate penyusutan 27 varian, serta konversi roasted-ke-green dan buffer tanpa file persediaan.

## Batasan

- Fitur tanggal kembar/gajian/akhir tahun adalah proksi kalender, bukan catatan promosi aktual. Dampak promosi hanya dapat divalidasi dengan kalender kampanye yang benar.
- Sebagian hubungan nama SKU/varian ke nama pada tabel penyusutan masih berstatus perlu konfirmasi. Estimasi green bean untuk pemetaan tersebut ditandai sementara.
- Data evaluasi 2024–2025 bersifat historis. Prediksi operasional 2026 memerlukan ekspor penjualan terbaru.
- Unggah hanya data yang dibutuhkan; kolom pribadi pelanggan tidak digunakan oleh model.
- Buffer berbasis kuantil kesalahan historis membantu perencanaan, tetapi tidak menjamin ketersediaan stok. Sampel kalibrasi validasi masih terbatas dan perlu diperbarui ketika data baru yang valid tersedia.
