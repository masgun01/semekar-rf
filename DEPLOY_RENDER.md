# Deploy Semekar RF ke Render (gratis untuk demo)

## Yang disiapkan
- `gunicorn` sudah ditambahkan ke `requirements.txt` agar aplikasi Flask dapat dijalankan pada server.
- `.python-version` mengarahkan Render menggunakan Python 3.12.
- File dataset tetap berada di `data/`; jangan hapus folder tersebut.

## Pengaturan Render
- Runtime: Python 3
- Root Directory: kosongkan (app.py dan requirements.txt ada di root repo)
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 240`
- Instance Type: Free

## Environment variables (wajib)
Tambahkan di Render > service > Environment:
- `APP_USER` = nama pengguna baru pilihan Anda
- `APP_PASS` = password panjang dan unik (jangan gunakan password bawaan)
- `SECRET_KEY` = string acak yang panjang

Untuk membuat SECRET_KEY lokal dari PowerShell, jalankan:
`py -c "import secrets; print(secrets.token_hex(32))"`
Salin hasilnya ke Render. Jangan masukkan rahasia ke GitHub. `GEMINI_API_KEY` hanya diperlukan bila fitur analisis AI eksternal ingin dipakai.

## Keamanan
Aplikasi memiliki kredensial bawaan `admin` / `semekar2026` bila `APP_USER` dan `APP_PASS` tidak disetel. WAJIB mengatur `APP_USER`, `APP_PASS`, dan `SECRET_KEY` sebelum membagikan URL publik. Jangan commit file `.env` atau API key.

## Catatan paket gratis Render
Layanan akan tidur setelah 15 menit tidak ada aktivitas, jadi kunjungan berikutnya dapat lambat saat bangun. Filesystem layanan gratis bersifat sementara: cache penelitian dapat dibuat ulang, tetapi file model operasional yang disimpan setelah pembaruan model dapat hilang saat layanan tidur/restart/redeploy. Gunakan ini untuk demo/presentasi; jangan jadikan satu-satunya penyimpanan operasional bisnis.
