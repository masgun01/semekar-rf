"""Prediksi kebutuhan stok roasted & green bean (Random Forest) - PT Semekar Kreatif Nusantara.
Jalankan:  pip install -r requirements.txt   lalu   python app.py   ->  http://127.0.0.1:5000
Login awal: admin / semekar2026  (ubah lewat variabel lingkungan APP_USER, APP_PASS, SECRET_KEY).
Analisis AI (opsional): isi GEMINI_API_KEY (Google AI Studio) atau ANTHROPIC_API_KEY, lewat variabel lingkungan atau file .env (lihat .env.example).
Jangan menulis kunci di kode dan jangan menyertakan .env saat mengumpulkan proyek. Tanpa kunci, analisis memakai aturan otomatis dan tetap jalan tanpa internet."""
import difflib, functools, gc, io, json, os, re, subprocess, sys, tempfile, urllib.error, urllib.request
from statistics import NormalDist
import joblib, numpy as np, pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import holidays
    HOL = holidays.country_holidays("ID", years=range(2020, 2041))
except Exception:
    HOL = {}

BASE = os.path.dirname(os.path.abspath(__file__))


def muat_env():
    """Baca file .env (opsional) di folder aplikasi: satu baris KUNCI=nilai. Variabel lingkungan yang sudah ada tidak ditimpa."""
    p = os.path.join(BASE, ".env")
    if os.path.exists(p):
        for baris_ in open(p, encoding="utf-8"):
            baris_ = baris_.strip()
            if baris_ and not baris_.startswith("#") and "=" in baris_:
                k, v = baris_.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


muat_env()
XLSX = os.path.join(BASE, "data", "Dataset_Sampling_v2.xlsx")
STOCK_XLSX = os.path.join(BASE, "data", "Stock.xlsx")
CACHE = os.path.join(BASE, "model_cache.joblib")
OP_CACHE = os.path.join(BASE, "operational_model.joblib")
MODEL_VERSION = "semekar-rf-2026-10-v23-validation-buffer-ux"
TRAIN_END = pd.Timestamp("2025-06-30")
VALID_START, VALID_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30")
TEST_START, TEST_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-24")
HZ_UJI = 30
SIZE = r"(\d+(?:[.,]\d+)?)\s*(kg|kilo|gram|gr|g)\b"
ALIAS = {"tanggal": ["tanggal", "date", "tgl", "waktu pesanan dibuat", "order creation time", "created time", "order date"],
         "varian": ["varian", "variant", "nama varian", "nama produk", "product name", "produk", "product", "sku"],
         "unit": ["terjual_unit", "jumlah", "quantity", "qty", "kuantitas", "jumlah produk", "unit"],
         "gram": ["gram_terjual", "gram", "berat"],
         "sku": ["nomor referensi sku", "sku", "kode sku", "seller sku", "sku id", "sku induk"],
         "variasi": ["nama variasi", "variation name", "variasi", "variation", "sku name", "ukuran"],
         "pesanan": ["no pesanan", "order id", "order number", "nomor pesanan", "id pesanan"],
         "retur": ["returned quantity", "jumlah retur", "return quantity", "quantity returned", "sku quantity of return"],
         "jenis": ["cancelation return type", "cancellation return type"],
         "status": ["status pesanan", "order status", "status"]}
UKURAN_UMUM = (100, 200, 500, 1000)
KOSONG = {"", "nan", "none", "null", "-", "--", "0", "n/a", "na", "tidak ada", "tanpa nama", "unknown"}
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _parse_shrinkage_rate(value):
    """Konversi angka penyusutan pada workbook ke fraksi 0..1.

    Nilai seperti 0.15 berarti 15%; teks '23%%' dari workbook juga
    diperlakukan sebagai 23%. Nilai di luar rentang fisik ditolak.
    """
    if pd.isna(value):
        return None
    raw = str(value).strip()
    has_percent = "%" in raw
    cleaned = raw.replace("%", "").replace(",", ".").strip()
    try:
        rate = float(cleaned)
    except (TypeError, ValueError):
        return None
    if has_percent or rate > 1:
        rate /= 100.0
    if not np.isfinite(rate) or rate < 0 or rate >= 1:
        return None
    return float(rate)


def load_variant_shrinkage(reference_xlsx=XLSX, stock_xlsx=STOCK_XLSX):
    """Bangun faktor green/roasted dari Stock.xlsx sheet Penyusutan.

    Nama di Referensi_Varian!nama_di_tabel_penyusutan menjadi jembatan
    eksplisit antara varian penjualan dan tabel penyusutan. Faktor dihitung
    dengan green = roasted / (1 - penyusutan). Status kecocokan nama tetap
    disimpan terpisah: status yang belum terkonfirmasi tidak menghapus
    estimasi green, tetapi harus ditampilkan sebagai estimasi sementara.
    """
    shrink_by_name = {}
    source_notes = []
    if os.path.exists(stock_xlsx):
        raw = pd.read_excel(stock_xlsx, sheet_name="Penyusutan", header=None)
        if raw.shape[0] < 3 or raw.shape[1] < 2:
            raise ValueError("Sheet 'Penyusutan' pada Stock.xlsx tidak memiliki bentuk tabel yang diharapkan.")
        conflicts = set()
        for col in range(1, raw.shape[1]):
            rate = _parse_shrinkage_rate(raw.iat[1, col])
            if rate is None:
                continue
            for row_idx in range(2, raw.shape[0]):
                value = raw.iat[row_idx, col]
                if pd.isna(value) or not str(value).strip():
                    continue
                name = str(value).strip()
                key = norm(name)
                if not key:
                    continue
                if key in shrink_by_name and not np.isclose(shrink_by_name[key]["penyusutan"], rate):
                    conflicts.add(key)
                else:
                    shrink_by_name[key] = {"nama_sumber_penyusutan": name, "penyusutan": rate}
        for key in conflicts:
            shrink_by_name.pop(key, None)
            source_notes.append(f"Nama penyusutan ganda dengan rate berbeda dikecualikan: {key}.")
    else:
        source_notes.append("Stock.xlsx tidak ditemukan; nilai fallback dari Referensi_Varian dipakai jika tersedia.")

    ref = pd.read_excel(reference_xlsx, sheet_name="Referensi_Varian").dropna(subset=["varian_acuan"]).copy()
    factors, losses, statuses, source_names = {}, {}, {}, {}
    missing = []
    for _, row in ref.iterrows():
        variant = str(row["varian_acuan"]).strip()
        lookup_name = str(row.get("nama_di_tabel_penyusutan", "")).strip()
        match = shrink_by_name.get(norm(lookup_name)) if lookup_name and norm(lookup_name) else None
        if match is not None:
            loss = float(match["penyusutan"])
            source_names[variant] = match["nama_sumber_penyusutan"]
        else:
            # Fallback transparan hanya bila workbook sumber tidak dapat memetakan nama.
            loss = _parse_shrinkage_rate(row.get("penyusutan"))
            if loss is None:
                missing.append(f"{variant} (nama penyusutan: {lookup_name or 'kosong'})")
                continue
            source_names[variant] = "Referensi_Varian (fallback)"
        if not 0 <= loss < 1:
            missing.append(f"{variant} (rate penyusutan tidak valid)")
            continue
        losses[variant] = loss
        factors[variant] = 1.0 / (1.0 - loss)
        status = str(row.get("kecocokan_nama", "Perlu konfirmasi perusahaan")).strip()
        statuses[variant] = status if status and status.lower() != "nan" else "Perlu konfirmasi perusahaan"

    audit = [f"Faktor green/roasted dihitung dari {os.path.basename(stock_xlsx)} sheet 'Penyusutan' untuk {len([x for x in source_names.values() if x != 'Referensi_Varian (fallback)']):,} varian; formula: green = roasted / (1 - penyusutan)."]
    if any(v == "Referensi_Varian (fallback)" for v in source_names.values()):
        audit.append("Sebagian faktor memakai fallback Referensi_Varian karena nama tidak ditemukan pada sheet Penyusutan; periksa daftar sumber faktor.")
    if missing:
        source_notes.append("Varian tanpa rate penyusutan: " + "; ".join(missing))
    audit.extend(source_notes)
    return {"faktor": factors, "penyusutan": losses, "pemetaan_status": statuses,
            "nama_penyusutan_sumber": source_names, "audit_penyusutan": audit, "jumlah_rate": len(losses)}


HIST_FEATS = ["lag_1", "lag_7", "lag_14", "lag_28", "rm_7", "rm_14", "rm_28", "std_7", "std_28", "rz_28"]
CAL_FEATS = ["dow", "dom", "bulan", "pekan", "day_sin", "day_cos", "month_sin", "month_cos", "akhir_pekan", "akhir_bulan", "libur"]
EVENT_FEATS = ["gajian", "tgl_kembar", "kampanye_9_9_12_12", "akhir_tahun"]
FEATURE_CONFIGS = [
    {"key": "historis", "name": "RF — fitur historis", "features": HIST_FEATS, "clip": False},
    {"key": "kalender", "name": "RF + kalender", "features": HIST_FEATS + CAL_FEATS, "clip": False},
    {"key": "kampanye", "name": "RF + kalender dan proksi kampanye", "features": HIST_FEATS + CAL_FEATS + EVENT_FEATS, "clip": False},
    {"key": "clipping", "name": "RF + kalender/kampanye (target clipping p99,5)", "features": HIST_FEATS + CAL_FEATS + EVENT_FEATS, "clip": True},
]


def _day_is_holiday(x):
    try:
        return int(x in HOL)
    except Exception:
        return 0


def add_features(df, target_col="gram_feature"):
    """Fitur yang tersedia pada tanggal t hanya memakai target sampai t-1; baris imputasi tetap NaN pada sejarah."""
    df = df.copy().sort_values(["varian", "tanggal"]).reset_index(drop=True)
    d = pd.to_datetime(df["tanggal"])
    df["dow"] = d.dt.dayofweek.astype(int)
    df["dom"] = d.dt.day.astype(int)
    df["bulan"] = d.dt.month.astype(int)
    df["pekan"] = d.dt.isocalendar().week.astype(int)
    df["day_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["bulan"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["bulan"] - 1) / 12)
    df["akhir_pekan"] = (df["dow"] >= 5).astype(int)
    days_to_month_end = (d.dt.days_in_month - d.dt.day)
    df["akhir_bulan"] = (days_to_month_end <= 3).astype(int)
    df["libur"] = d.dt.date.map(_day_is_holiday).astype(int)
    df["gajian"] = ((df["dom"] >= 25) | (df["dom"] <= 3)).astype(int)
    df["tgl_kembar"] = (df["dom"] == df["bulan"]).astype(int)
    df["kampanye_9_9_12_12"] = ((df["bulan"].isin([9, 10, 11, 12])) & (df["dom"] == df["bulan"])).astype(int)
    df["akhir_tahun"] = ((df["bulan"] == 12) & (df["dom"] >= 15)).astype(int)

    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = df.groupby("varian", sort=False)[target_col].shift(lag)
    for window in (7, 14, 28):
        df[f"rm_{window}"] = df.groupby("varian", sort=False)[target_col].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
    for window in (7, 28):
        df[f"std_{window}"] = df.groupby("varian", sort=False)[target_col].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=2).std(ddof=0))
    def zero_rate(s):
        prior = s.shift(1)
        zero = prior.eq(0).astype(float).where(prior.notna())
        return zero.rolling(28, min_periods=1).mean()
    df["rz_28"] = df.groupby("varian", sort=False)[target_col].transform(zero_rate)
    return df


def metrik(y, p):
    a, pred = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ok = np.isfinite(a) & np.isfinite(pred)
    a, pred = a[ok], pred[ok]
    if not len(a):
        return {"MAE": None, "RMSE": None, "MAPE": None, "WAPE": None, "R2": None, "n": 0}
    err = a - pred
    nz = a > 0
    total = float(a.sum())
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err[nz] / a[nz])) * 100) if nz.any() else None,
        "WAPE": float(np.abs(err).sum() / total * 100) if total > 0 else None,
        "R2": float(r2_score(a, pred)) if len(a) > 1 and np.unique(a).size > 1 else None,
        "n": int(len(a)),
    }


def rf():
    """Random Forest dengan identitas varian one-hot; fitur numerik diimputasi di dalam pipeline."""
    return RandomForestRegressor(n_estimators=60, min_samples_leaf=4, max_features=0.75,
                                 n_jobs=1, random_state=42)


def _build_estimator(features):
    prep = ColumnTransformer([
        ("numerik", SimpleImputer(strategy="median", keep_empty_features=True), list(features)),
        ("varian", OneHotEncoder(handle_unknown="ignore"), ["varian"]),
    ], remainder="drop")
    return Pipeline([("prep", prep), ("rf", rf())])


def _eligible_frame(f):
    return f["label_valid"].fillna(False).astype(bool) & f["layak_riwayat"].fillna(False).astype(bool)


def _fit_model(tr, config):
    cols = list(config["features"])
    eligible = _eligible_frame(tr) & pd.to_numeric(tr["gram_terjual"], errors="coerce").notna()
    train_df = tr.loc[eligible].copy()
    if len(train_df) < 100:
        raise ValueError(f"Data latih terlalu sedikit setelah mengecualikan target imputasi: {len(train_df)} baris.")
    y = train_df["gram_terjual"].astype(float).to_numpy()
    clip_cap = None
    if config.get("clip"):
        clip_cap = float(np.quantile(y, 0.995))
        y = np.minimum(y, clip_cap)
    model = _build_estimator(cols)
    # Batasi native BLAS/OpenMP threads. Ini membuat fitting berulang dan forecast
    # rekursif stabil pada workstation dengan jumlah CPU terbatas.
    with threadpool_limits(limits=1):
        model.fit(train_df[cols + ["varian"]], y)
    return model, clip_cap


def _predict(model, data, features):
    if data.empty:
        return np.asarray([], dtype=float)
    with threadpool_limits(limits=1):
        pred = model.predict(data[list(features) + ["varian"]])
    return np.clip(np.asarray(pred, dtype=float), 0, None)


def _create_research_frame():
    if not os.path.exists(XLSX):
        raise FileNotFoundError("Dataset_Sampling_v2.xlsx tidak ditemukan di folder data/.")
    h = pd.read_excel(XLSX, sheet_name="Data_Harian")
    required = {"tanggal", "varian", "gram_terjual"}
    missing = required - set(h.columns)
    if missing:
        raise ValueError("Kolom wajib tidak ada pada Data_Harian: " + ", ".join(sorted(missing)))
    audit = []
    h["tanggal"] = pd.to_datetime(h["tanggal"], errors="coerce").dt.normalize()
    h["varian"] = h["varian"].astype(str).str.strip()
    h["gram_terjual"] = pd.to_numeric(h["gram_terjual"], errors="coerce")
    if "terjual_unit" not in h:
        h["terjual_unit"] = np.nan
    h["terjual_unit"] = pd.to_numeric(h["terjual_unit"], errors="coerce")
    if "imputasi" not in h:
        h["imputasi"] = 0
        audit.append("Kolom imputasi tidak ditemukan; baris tidak ditandai imputasi secara otomatis. Verifikasi sumber data.")
    h["imputasi"] = pd.to_numeric(h["imputasi"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    bad_dates = int(h["tanggal"].isna().sum())
    if bad_dates:
        audit.append(f"{bad_dates} baris tanggal tidak valid dikeluarkan.")
        h = h[h["tanggal"].notna()].copy()
    neg = int((h["gram_terjual"] < 0).sum())
    if neg:
        audit.append(f"{neg} target gram negatif ditandai tidak valid (bukan diubah menjadi nol).")
        h.loc[h["gram_terjual"] < 0, "gram_terjual"] = np.nan
    h["varian"] = h["varian"].replace({"nan": np.nan, "": np.nan})
    h = h.dropna(subset=["tanggal", "varian"])
    duplicate_n = int(h.duplicated(["tanggal", "varian"], keep=False).sum())
    if duplicate_n:
        audit.append(f"Terdapat {duplicate_n} baris pada kunci tanggal-varian yang berulang; digabung agar satu baris per tanggal-varian.")
        h = h.groupby(["tanggal", "varian"], as_index=False).agg(
            terjual_unit=("terjual_unit", lambda x: x.sum(min_count=1)),
            gram_terjual=("gram_terjual", lambda x: x.sum(min_count=1)),
            imputasi=("imputasi", "max"))
    # Jika ada satu baris imputasi di satu tanggal, perlakukan seluruh tanggal sebagai tidak lengkap.
    incomplete_dates = set(h.loc[h["imputasi"].eq(1), "tanggal"].dt.normalize())
    if incomplete_dates:
        audit.append(f"{len(incomplete_dates)} tanggal terindikasi tidak lengkap/imputasi dikeluarkan dari label latih/evaluasi; nilai tanggal tersebut tidak dipakai sebagai fitur lag.")
    parts = []
    for v, g in h.groupby("varian", sort=False):
        g = g.sort_values("tanggal").set_index("tanggal")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        has_row = pd.Series(idx.isin(g.index), index=idx)
        gg = g.reindex(idx)
        gg.index.name = "tanggal"
        gg["varian"] = v
        gg["imputasi"] = gg["imputasi"].fillna(1).astype(int)
        # Gap tanggal tidak dianggap sebagai nol penjualan yang pasti.
        missing_rows = int((~has_row).sum())
        if missing_rows:
            audit.append(f"Varian {v}: {missing_rows} tanggal tidak memiliki baris sumber; target dibiarkan hilang, bukan diisi nol.")
        parts.append(gg.reset_index())
    f = pd.concat(parts, ignore_index=True).sort_values(["varian", "tanggal"]).reset_index(drop=True)
    # Jika gap pada satu varian membuktikan tanggal tidak lengkap, kecualikan tanggal itu untuk semua varian.
    incomplete_dates.update(f.loc[f["imputasi"].eq(1), "tanggal"].dropna().dt.normalize().tolist())
    if "fase" not in f:
        f["fase"] = np.select([f["tanggal"] <= TRAIN_END, f["tanggal"] <= VALID_END, f["tanggal"] <= TEST_END],
                               ["latih", "validasi", "uji"], default="operasional")
    else:
        f["fase"] = np.select([f["tanggal"] <= TRAIN_END, f["tanggal"] <= VALID_END, f["tanggal"] <= TEST_END],
                               ["latih", "validasi", "uji"], default="operasional")
    f["tanggal_tidak_lengkap"] = f["tanggal"].isin(incomplete_dates)
    f["label_valid"] = (~f["tanggal_tidak_lengkap"]) & f["gram_terjual"].notna() & (f["gram_terjual"] >= 0)
    # Angka imputasi tidak masuk fitur lag/rolling; target aktual aslinya tetap tersedia untuk audit.
    f["gram_feature"] = f["gram_terjual"].where(~f["tanggal_tidak_lengkap"])
    f["layak_riwayat"] = f.groupby("varian", sort=False).cumcount().ge(28)
    f = add_features(f)
    audit.append(f"Dataset: {len(f):,} baris varian-hari, {f['varian'].nunique()} varian, {int(f['label_valid'].sum()):,} target aktual layak, {int((~f['label_valid']).sum()):,} target tidak valid/imputasi.")
    return f, audit


def _backtest(f, config, start, end, capture_models=False):
    """Fit pada data sebelum cutoff, lalu evaluasi one-step-ahead pada rentang berikutnya.

    Fitur lag untuk setiap tanggal hanya memakai penjualan sampai hari sebelumnya. Dengan
    demikian, selama validasi/uji, nilai aktual hari sebelumnya boleh menjadi riwayat yang
    diketahui, tetapi target tanggal yang sedang diprediksi tidak masuk ke fit model.
    Evaluasi horizon 7/14/30 hari tanpa aktual masa depan dilakukan terpisah secara rekursif.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    tr = f[f["tanggal"] < start]
    te = f[(f["tanggal"] >= start) & (f["tanggal"] <= end) & _eligible_frame(f)]
    if tr.empty or te.empty:
        return pd.DataFrame(columns=["tanggal", "varian", "gram_terjual", "prediksi", "blok_latih_mulai"]), {}
    model, _ = _fit_model(tr, config)
    pred = _predict(model, te, config["features"])
    out = te[["tanggal", "varian", "gram_terjual"]].copy()
    out["prediksi"] = pred
    out["blok_latih_mulai"] = start
    fitted = {start: model} if capture_models else {}
    if not capture_models:
        del model
        gc.collect()
    return out.reset_index(drop=True), fitted


def _metric_dict(d):
    return metrik(d["gram_terjual"], d["prediksi"]) if len(d) else metrik([], [])


def _feature_importance(model):
    try:
        names = model.named_steps["prep"].get_feature_names_out()
        values = model.named_steps["rf"].feature_importances_
        combined = {}
        for name, value in zip(names, values):
            if name.startswith("numerik__"):
                label = name.replace("numerik__", "")
            elif name.startswith("varian__"):
                label = "Identitas varian (one-hot)"
            else:
                label = name
            combined[label] = combined.get(label, 0.0) + float(value)
        return sorted(((k, round(v, 4)) for k, v in combined.items()), key=lambda x: -x[1])[:20]
    except Exception:
        return []


def metrik_rek(g, tl):
    a, p = g["gram_terjual"].to_numpy(float), g["prediksi"].to_numpy(float)
    d = a - p
    total = float(a.sum())
    return {"akurasi": max(0.0, 100 - float(np.abs(d).sum() / max(total, 1) * 100)),
            "galat_kg": float(np.abs(d).mean() / 1000) if len(d) else None,
            "bias": float(d.sum() / max(total, 1) * 100),
            "buffer": float(np.quantile(np.clip(d / np.clip(p, 1, None), 0, None), tl)) * 100 if len(d) else np.nan}


def forecast(feat, B, hz):
    """Ramalan rekursif: setiap prediksi yang dibuat menjadi riwayat berikutnya."""
    last = pd.to_datetime(feat["tanggal"]).max()
    source_col = "gram_feature" if "gram_feature" in feat else "gram_terjual"
    # Semua fitur historis maksimum memakai 28 hari; memuat 60 hari lalu membangun
    # ulang seluruh fitur pada setiap langkah hanya menambah biaya komputasi.
    h = feat.loc[feat["tanggal"] >= last - pd.Timedelta(days=27), ["tanggal", "varian", source_col]].copy()
    h = h.rename(columns={source_col: "gram_feature"})
    variants = h["varian"].dropna().unique()
    rows = []
    for k in range(1, hz + 1):
        d = last + pd.Timedelta(days=k)
        h = pd.concat([h, pd.DataFrame({"tanggal": [d] * len(variants), "varian": variants, "gram_feature": np.nan})], ignore_index=True)
        x = add_features(h, target_col="gram_feature")
        x = x[x["tanggal"] == d].copy()
        x["prediksi"] = _predict(B["model"], x, B["features"])
        predicted = dict(zip(x["varian"], x["prediksi"]))
        mask = h["tanggal"] == d
        h.loc[mask, "gram_feature"] = h.loc[mask, "varian"].map(predicted)
        rows.append(x[["tanggal", "varian", "prediksi"]])
        # Pertahankan hanya 28 hari terbaru untuk langkah rekursif berikutnya.
        h = h[h["tanggal"] >= d - pd.Timedelta(days=27)].copy()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["tanggal", "varian", "prediksi"])


def _recursive_backtest(f, config, models_by_block, tmax, origin_start=None, horizon_max=30):
    """Evaluasi ramalan rekursif dari origin yang ditetapkan.

    origin_start memungkinkan pemisahan eksplisit antara backtest penelitian
    (uji) dan kalibrasi buffer (validasi). Buffer tidak boleh dikalibrasi dari
    periode test final.
    """
    rows = []
    origin_start = pd.Timestamp(origin_start if origin_start is not None else TEST_START)
    tmax = pd.Timestamp(tmax)
    horizon_max = max(7, min(int(horizon_max), 90))
    origins = pd.date_range(origin_start, max(origin_start, tmax - pd.Timedelta(days=horizon_max)), freq="7D")
    block_starts = sorted(models_by_block)
    complete_dates = set(f.loc[~f["tanggal_tidak_lengkap"], "tanggal"].drop_duplicates())
    for o in origins:
        eligible_blocks = [b for b in block_starts if b <= o]
        if not eligible_blocks:
            continue
        b = eligible_blocks[-1]
        model = models_by_block[b]
        hz = min(horizon_max, (tmax - o).days)
        if hz < 7:
            continue
        f_hist = f[f["tanggal"] <= o].copy()
        pred = forecast(f_hist, {"model": model, "features": config["features"]}, hz)
        truth = f[(f["tanggal"] > o) & (f["tanggal"] <= o + pd.Timedelta(days=hz)) &
                  (f["tanggal"].isin(complete_dates)) & f["gram_terjual"].notna()][["tanggal", "varian", "gram_terjual"]]
        m = pred.merge(truth, on=["tanggal", "varian"], how="inner")
        for k in (7, 14, 30):
            if k > hz:
                continue
            # Tanggal imputasi dikeluarkan untuk semua varian, bukan dihitung sebagai nol.
            z = m[m["tanggal"] <= o + pd.Timedelta(days=k)]
            if z.empty:
                continue
            agg = z.groupby("varian", as_index=False).agg(prediksi=("prediksi", "sum"), gram_terjual=("gram_terjual", "sum"))
            agg["o"], agg["hz"] = o, k
            rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["varian", "prediksi", "gram_terjual", "o", "hz"])


def tabel_level(B):
    def sat(g):
        a, p = g["gram_terjual"].to_numpy(float), g["prediksi"].to_numpy(float)
        total = float(a.sum())
        return {"a": float(np.mean(a) / 1000) if len(a) else None,
                "p": float(np.mean(p) / 1000) if len(p) else None,
                "galat": float(np.mean(np.abs(a - p)) / 1000) if len(a) else None,
                "akurasi": max(0.0, 100 - float(np.abs(a - p).sum() / max(total, 1) * 100)) if len(a) else None}
    out = {}
    levels = [("1", B["bt"], "tanggal")] + [(str(k), B["rek"][B["rek"]["hz"] == k], "o") for k in (7, 14, 30)]
    for k, d, key in levels:
        if d.empty:
            out[k] = []
            continue
        all_rows = [{"varian": "SEMUA VARIAN", **sat(d.groupby(key)[["gram_terjual", "prediksi"]].sum().reset_index())}]
        all_rows += [{"varian": v, **sat(g)} for v, g in d.groupby("varian")]
        out[k] = all_rows
    return out


def _score_rows(d, actual_col="gram_terjual", pred_col="prediksi", aggregate_key=None):
    if d.empty:
        return {"total": None, "varian": None, "median": None}
    a, p = d[actual_col].to_numpy(float), d[pred_col].to_numpy(float)
    row_wape = float(np.abs(a-p).sum()/a.sum()*100) if a.sum() > 0 else None
    grouped = d.groupby("varian", sort=False).apply(
        lambda g: float(np.abs(g[actual_col]-g[pred_col]).sum()/g[actual_col].sum()*100) if g[actual_col].sum() > 0 else np.nan)
    vals = pd.to_numeric(grouped, errors="coerce").dropna()
    if aggregate_key and aggregate_key in d:
        total_df = d.groupby(aggregate_key, as_index=False)[[actual_col, pred_col]].sum()
        den = float(total_df[actual_col].sum())
        agg_wape = float(np.abs(total_df[actual_col]-total_df[pred_col]).sum()/den*100) if den > 0 else None
    else:
        agg_wape = row_wape
    return {"total": round(max(0.0, 100-agg_wape), 1) if agg_wape is not None else None,
            "varian": round(max(0.0, 100-row_wape), 1) if row_wape is not None else None,
            "median": round(max(0.0, 100-float(vals.median())), 1) if len(vals) else None}


def banding(test_all, rek, model_selection, f):
    methods = []
    baseline_specs = [("Baseline: lag 7 hari", "lag_7"), ("Baseline: rata-rata 7 hari", "rm_7"), ("Baseline: rata-rata 28 hari", "rm_28")]
    config_columns = {
        "historis": "pred_historis", "kalender": "pred_kalender", "kampanye": "pred_kampanye", "clipping": "pred_clipping"
    }
    common = test_all.copy()
    all_pred_cols = [col for _, col in baseline_specs] + list(config_columns.values())
    # Semua metode dibandingkan pada pasangan varian-tanggal identik. Baris yang tidak
    # mempunyai sejarah cukup untuk baseline dikeluarkan dari seluruh perbandingan ini.
    shared = common.dropna(subset=[c for c in all_pred_cols if c in common.columns]).copy()
    print(f"Banding harian: {len(shared):,} baris bersama.")
    for name, col in baseline_specs:
        if col in shared:
            shared[name] = np.clip(pd.to_numeric(shared[col], errors="coerce"), 0, None)
            methods.append({"metode": name, **_score_rows(shared, "gram_terjual", name, aggregate_key="tanggal")})
    for c in FEATURE_CONFIGS:
        col = config_columns[c["key"]]
        if col in shared:
            methods.append({"metode": c["name"], **_score_rows(shared.rename(columns={col: "_p"}), "gram_terjual", "_p", aggregate_key="tanggal")})
    print("Banding harian selesai.")
    daily = {"tingkat": "Harian", "ket": "Semua metode dibandingkan pada pasangan varian-tanggal yang sama. Baris pada tanggal yang ditandai imputasi tidak masuk metrik. Fitur kampanye adalah proksi kalender, bukan bukti promosi aktual.",
             "n_titik": int(shared["tanggal"].nunique()) if not shared.empty else 0,
             "n_baris": int(len(shared)), "metode": methods}
    recursive_levels = []
    for hz in (7, 30):
        print(f"Banding rekursif horizon {hz}: mulai.")
        r = rek[rek["hz"] == hz].copy()
        if r.empty:
            recursive_levels.append({"tingkat": f"{hz} hari", "ket": "Tidak cukup data aktual lengkap untuk evaluasi rekursif pada horizon ini.", "n_titik": 0, "n_baris": 0, "metode": []})
            continue
        baseline_rows = []
        for row in r.itertuples(index=False):
            hist = f[(f["varian"] == row.varian) & (f["tanggal"] <= row.o) & f["gram_feature"].notna()].sort_values("tanggal")
            vals7, vals28 = hist["gram_feature"].tail(7), hist["gram_feature"].tail(28)
            baseline_rows.append({"varian": row.varian, "o": row.o, "gram_terjual": row.gram_terjual,
                                  "model": row.prediksi,
                                  "rata7": float(vals7.mean() * hz) if len(vals7) else np.nan,
                                  "rata28": float(vals28.mean() * hz) if len(vals28) else np.nan})
        q = pd.DataFrame(baseline_rows)
        print(f"Banding rekursif horizon {hz}: baseline siap ({len(q):,} baris).")
        rec_methods = []
        for name, col in [(f"Random Forest terpilih: {model_selection['selected']}", "model"), ("Baseline: rata-rata 7 hari", "rata7"), ("Baseline: rata-rata 28 hari", "rata28")]:
            rec_methods.append({"metode": name, **_score_rows(q.dropna(subset=[col]).rename(columns={col: "_p"}), "gram_terjual", "_p", aggregate_key="o")})
        print(f"Banding rekursif horizon {hz}: selesai.")
        recursive_levels.append({"tingkat": f"{hz} hari", "ket": f"Ramalan rekursif {hz} hari dari tiap titik asal; hasil ramalan sebelumnya dipakai sebagai riwayat. Titik asal berjarak 7 hari. Model tetap dibekukan pada cutoff training 30 Juni 2025 selama rangkaian uji agar periode validasi/test tidak masuk pelatihan. Tanggal sumber tidak lengkap dikeluarkan.",
                                "n_titik": int(r["o"].nunique()), "n_baris": int(len(r)), "metode": rec_methods})
    return {"uji": f"{TEST_START.date()} s.d. {TEST_END.date()}", "pemilihan": f"Konfigurasi dipilih berdasarkan WAPE validasi {VALID_START.date()} s.d. {VALID_END.date()}: {model_selection['selected']}.",
            "tingkat": [daily] + recursive_levels}


def kesimpulan_banding(bd):
    isi = [bd.get("pemilihan", "Konfigurasi dipilih berdasarkan validasi temporal.")]
    for t in bd.get("tingkat", []):
        valid = [m for m in t.get("metode", []) if m.get("total") is not None]
        if not valid:
            isi.append(f"{t['tingkat']}: tidak ada cukup data valid untuk membandingkan metode.")
            continue
        best_total = max(valid, key=lambda m: m["total"])
        med = [m for m in valid if m.get("median") is not None]
        best_median = max(med, key=lambda m: m["median"]) if med else best_total
        isi.append(f"{t['tingkat']}: metode terbaik berdasarkan akurasi total adalah {best_total['metode']} ({best_total['total']:.1f}%); berdasarkan median per varian adalah {best_median['metode']} ({best_median.get('median', best_median['total']):.1f}%). Nilai tersebut hasil pengujian, bukan jaminan kinerja periode mendatang.")
    isi.append("Catatan: fitur tanggal kembar, tanggal gajian, dan akhir tahun adalah proksi kalender. Kalender promosi perusahaan belum tersedia, sehingga dampak diskon/promosi aktual belum dapat dipastikan.")
    return isi


def _fit_operational_isolated(config_key):
    """Fit model operasional di proses Python terpisah agar state threadpool backtest tidak terbawa."""
    fd, output_path = tempfile.mkstemp(prefix="semekar-operasional-", suffix=".joblib", dir=BASE)
    os.close(fd)
    os.unlink(output_path)
    cmd = [sys.executable, os.path.join(BASE, "app.py"), "--fit-operational", config_key, output_path]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(cmd, cwd=BASE, env=env, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0 or not os.path.exists(output_path):
            details = (proc.stderr or proc.stdout or "tidak ada detail dari proses pelatihan").strip()
            raise RuntimeError("Pelatihan model operasional gagal pada proses terpisah: " + details[-1800:])
        artifact = joblib.load(output_path)
        if artifact.get("version") != MODEL_VERSION or "model" not in artifact:
            raise RuntimeError("Artefak model operasional tidak valid atau berbeda versi.")
        return artifact["model"], artifact.get("cap")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Pelatihan model operasional melewati batas proses. Coba lagi setelah memastikan sumber daya komputer tersedia.") from exc
    finally:
        try:
            if os.path.exists(output_path): os.remove(output_path)
        except OSError:
            pass


def _fit_operational_cli(config_key, output_path):
    cfg = next((c for c in FEATURE_CONFIGS if c["key"] == config_key), None)
    if cfg is None:
        raise ValueError("Kunci konfigurasi model tidak dikenal: " + str(config_key))
    f, _ = _create_research_frame()
    valid_label = f["label_valid"] & f["layak_riwayat"]
    operational_train = f[valid_label].copy()
    model, cap = _fit_model(operational_train, cfg)
    joblib.dump({"version": MODEL_VERSION, "model": model, "cap": cap}, output_path, compress=3)


def train():
    print("Audit data dan menjalankan backtest temporal Semekar...")
    f, audit = _create_research_frame()
    print(f"Dataset siap: {len(f):,} baris; mulai validasi temporal.")
    valid_label = f["label_valid"] & f["layak_riwayat"]
    train_y = f.loc[valid_label & (f["tanggal"] <= TRAIN_END), "gram_terjual"].astype(float)
    if train_y.empty:
        raise ValueError("Tidak ada data latih aktual yang valid sampai 30 Juni 2025.")
    cap = float(train_y.quantile(0.995))
    validation = {}
    validation_models = {}
    for cfg in FEATURE_CONFIGS:
        # Simpan model yang dilatih hanya sampai 30 Juni; model yang sama akan dievaluasi
        # pada test agar konfigurasi dibandingkan pada cutoff pelatihan identik.
        pred, fitted = _backtest(f, cfg, VALID_START, VALID_END, capture_models=True)
        validation[cfg["key"]] = pred
        validation_models[cfg["key"]] = fitted.get(VALID_START)
        print(f"Validasi {cfg['key']} selesai: {len(pred):,} prediksi.")
    valid_scores = []
    for cfg in FEATURE_CONFIGS:
        m = _metric_dict(validation[cfg["key"]])
        valid_scores.append({"key": cfg["key"], "nama": cfg["name"], **m})
    available_scores = [m for m in valid_scores if m["WAPE"] is not None]
    if not available_scores:
        raise ValueError("Backtest validasi tidak menghasilkan prediksi valid. Periksa rentang tanggal dan data.")
    selected_key = min(available_scores, key=lambda x: x["WAPE"])["key"]
    selected_cfg = next(c for c in FEATURE_CONFIGS if c["key"] == selected_key)
    print(f"Konfigurasi dipilih dari validasi: {selected_cfg['name']}.")
    test_preds, models_by_config = {}, {}
    test_rows = f[(f["tanggal"] >= TEST_START) & (f["tanggal"] <= TEST_END) & _eligible_frame(f)].copy()
    for cfg in FEATURE_CONFIGS:
        # Evaluasi test menggunakan model yang sama cutoff-nya (latih s.d. 30 Juni)
        # untuk semua konfigurasi. Parameter tidak dituning dari validasi atau test.
        model = validation_models.get(cfg["key"])
        if model is None:
            raise ValueError(f"Model validasi {cfg['key']} tidak tersedia untuk evaluasi test.")
        pred = _predict(model, test_rows, cfg["features"])
        q = test_rows[["tanggal", "varian", "gram_terjual"]].copy()
        q["prediksi"] = pred
        q["blok_latih_mulai"] = TRAIN_END + pd.Timedelta(days=1)
        test_preds[cfg["key"]] = q
        print(f"Prediksi uji satu-langkah {cfg['key']} selesai: {len(q):,} baris.")
    # Kalibrasi buffer harus berasal dari VALIDASI, bukan periode test final.
    # Model tetap model yang dilatih hanya sampai 30 Juni 2025; origin mingguan
    # memakai riwayat yang tersedia sampai origin, lalu memprediksi 7/14/30 hari.
    recursive_model = validation_models[selected_key]
    print("Menyusun kalibrasi buffer dari periode validasi (terpisah dari test)...")
    buffer_calibration = _recursive_backtest(
        f, selected_cfg, {VALID_START: recursive_model}, VALID_END,
        origin_start=VALID_START, horizon_max=30
    )
    print(f"Kalibrasi buffer validasi selesai: {len(buffer_calibration):,} baris.")

    # Evaluasi rekursif penelitian pada periode test dipisah dari kalibrasi buffer.
    del validation_models, validation, model, fitted, pred, q
    gc.collect()
    models_for_recursive = {TEST_START: recursive_model}
    tmax = f["tanggal"].max()
    print(f"Mulai backtest rekursif uji dengan cutoff model 30 Juni sampai {tmax.date()}.")
    recursive = _recursive_backtest(f, selected_cfg, models_for_recursive, tmax,
                                    origin_start=TEST_START, horizon_max=30)
    print(f"Backtest rekursif uji selesai: {len(recursive):,} baris.")
    del models_for_recursive, recursive_model
    gc.collect()
    base = test_preds[selected_key].copy()
    if base.empty:
        raise ValueError("Backtest periode uji tidak menghasilkan data. Pastikan dataset mencakup Oktober–Desember 2025.")
    test_all = base[["tanggal", "varian", "gram_terjual", "blok_latih_mulai"]].copy()
    colmap = {"historis": "pred_historis", "kalender": "pred_kalender", "kampanye": "pred_kampanye", "clipping": "pred_clipping"}
    for key, d in test_preds.items():
        q = d[["tanggal", "varian", "prediksi"]].rename(columns={"prediksi": colmap[key]})
        test_all = test_all.merge(q, on=["tanggal", "varian"], how="left", validate="one_to_one")
    # Baseline menggunakan sejarah yang tersedia sampai sehari sebelum target.
    feature_columns = list(set(HIST_FEATS + CAL_FEATS + EVENT_FEATS + ["tanggal", "varian", "gram_terjual", "gram_feature"]))
    f_cols = f[[c for c in feature_columns if c in f.columns]].copy()
    test_all = test_all.merge(f_cols[["tanggal", "varian"] + [c for c in ("lag_1", "lag_7", "rm_7", "rm_28") if c in f_cols]],
                              on=["tanggal", "varian"], how="left", validate="one_to_one")
    test_all = test_all.drop(columns=["blok_latih_mulai"], errors="ignore")
    test_selected = test_preds[selected_key].copy().rename(columns={"prediksi": "prediksi"})
    test_metrics = {cfg["key"]: {"key": cfg["key"], "nama": cfg["name"], **_metric_dict(test_preds[cfg["key"]])} for cfg in FEATURE_CONFIGS}
    model_selection = {"selected_key": selected_key, "selected": selected_cfg["name"],
                       "validation": valid_scores, "test": list(test_metrics.values()),
                       "test_excluded_imputed_rows": int((f["tanggal_tidak_lengkap"] & f["tanggal"].between(TEST_START, TEST_END)).sum()),
                       "method": "WAPE validasi minimum; test hanya untuk pelaporan. Perbandingan satu-langkah seluruh konfigurasi dan evaluasi rekursif memakai model yang dilatih sampai 30 Juni; test hanya untuk pelaporan."}
    # Model operasional dilatih setelah seluruh evaluasi historis selesai agar tidak
    # menahan dua forest besar sekaligus selama backtest rekursif.
    ref = pd.read_excel(XLSX, sheet_name="Referensi_Varian").dropna(subset=["varian_acuan"]).copy()
    print("Referensi varian dibaca.")
    all_known = ref["varian_acuan"].astype(str).str.strip().unique().tolist()
    shrinkage = load_variant_shrinkage()
    factors = shrinkage["faktor"]
    losses = shrinkage["penyusutan"]
    mapping_status = shrinkage["pemetaan_status"]
    shrinkage_names = shrinkage["nama_penyusutan_sumber"]
    print(f"Faktor green/roasted tersedia untuk {len(factors)} dari {len(all_known)} varian berdasarkan tabel penyusutan.")
    sku_map = {}
    for _, r in ref.dropna(subset=["kode_sku_penjualan"]).iterrows():
        for sku in str(r["kode_sku_penjualan"]).split(","):
            sku_map[sku.strip().upper()] = str(r["varian_acuan"]).strip()
    print(f"Pemetaan penyusutan dimuat: {len(factors)} faktor tersedia dari {len(all_known)} varian; status identitas produk disimpan terpisah.")
    print("Menyiapkan data latih operasional akhir.")
    operational_train = f[valid_label].copy()
    print(f"Melatih model operasional pada {len(operational_train):,} baris.")
    del operational_train
    final, final_cap = _fit_operational_isolated(selected_key)
    print("Model operasional dilatih ulang dalam proses terpisah setelah evaluasi historis selesai.")
    bt = test_selected[["tanggal", "varian", "gram_terjual", "prediksi"]].copy()
    global_metrics = metrik(bt["gram_terjual"], bt["prediksi"])
    pv = []
    for v, g in bt.groupby("varian"):
        actual, pred = float(g["gram_terjual"].sum()), float(g["prediksi"].sum())
        pv.append({"varian": v, "aktual_kg": round(actual / 1000, 3), "prediksi_kg": round(pred / 1000, 3),
                   "wape": round(float(np.abs(g["gram_terjual"] - g["prediksi"]).sum() / actual * 100), 1) if actual > 0 else None})
    # Jangan tampilkan agregat green bean parsial seolah mencakup semua varian.
    required_variants = set(bt["varian"].dropna().unique())
    if required_variants and required_variants.issubset(set(factors)):
        green_actual = float((bt["gram_terjual"] * bt["varian"].map(factors)).sum() / 1000)
        green_pred = float((bt["prediksi"] * bt["varian"].map(factors)).sum() / 1000)
    else:
        green_actual = None
        green_pred = None
    phases = []
    for phase, start, end in [("latih", None, TRAIN_END), ("validasi", VALID_START, VALID_END), ("uji", TEST_START, TEST_END)]:
        g = f[(f["tanggal"] <= end) if start is None else f["tanggal"].between(start, end)]
        phases.append({"fase": phase, "n": int((g["label_valid"] & g["layak_riwayat"]).sum()),
                       "n_imputasi": int(g["tanggal_tidak_lengkap"].sum()),
                       "dari": str(g["tanggal"].min().date()), "sampai": str(g["tanggal"].max().date())})
    print("Menghitung feature importance.")
    imp = _feature_importance(final)
    print("Feature importance selesai.")
    recursive_for_stats = recursive
    if recursive_for_stats.empty:
        raise ValueError("Evaluasi rekursif belum memiliki titik asal yang cukup untuk menghasilkan ukuran stok pengaman.")
    print("Menyusun tabel perbandingan model.")
    band = banding(test_all, recursive, model_selection, f)
    print("Ringkasan perbandingan selesai.")
    audit.append(f"Konfigurasi terpilih dari validasi (WAPE minimum): {selected_cfg['name']}.")
    audit.append("Perbandingan satu-langkah dan evaluasi rekursif 7/14/30 hari memakai model yang dilatih sampai 30 Juni 2025; model operasional akhir dilatih setelah evaluasi historis selesai.")
    audit.append("Clipping target p99,5 hanya salah satu konfigurasi eksperimen. Bila terpilih berdasarkan WAPE validasi, clipping diterapkan pada label pelatihan konfigurasi tersebut saja; evaluasi selalu memakai target aktual yang tidak dipangkas.")
    provisional_variants = [v for v in factors if not str(mapping_status.get(v, "")).lower().startswith("cocok")]
    if provisional_variants:
        audit.append(f"Estimasi green bean tersedia untuk seluruh varian yang memiliki rate penyusutan. Identitas SKU/varian untuk {len(provisional_variants)} varian masih berstatus perlu konfirmasi; tampilkan hasil sebagai estimasi sementara dan validasi sebelum pembelian.")
    return {"version": MODEL_VERSION, "model": final, "features": selected_cfg["features"], "selected_config": selected_cfg,
            "rek": recursive, "buffer_calibration": buffer_calibration,
            "banding": band, "cap": cap, "final_cap": final_cap, "faktor": factors,
            "penyusutan": losses, "pemetaan_status": mapping_status, "nama_penyusutan_sumber": shrinkage_names,
            "audit_penyusutan": shrinkage["audit_penyusutan"], "known": all_known, "sku": sku_map,
            "bt": bt, "test_all": test_all, "metrik": global_metrics, "stok": {"roasted_aktual": float(bt["gram_terjual"].sum()/1000),
            "roasted_prediksi": float(bt["prediksi"].sum()/1000), "green_aktual": green_actual, "green_prediksi": green_pred},
            "imp": imp, "split": phases, "pv": sorted(pv, key=lambda r: -r["aktual_kg"]),
            "selection": model_selection, "audit": audit}


_B = None

def bundle(force=False):
    global _B
    if _B is None or force:
        if not force and os.path.exists(CACHE):
            try:
                cached = joblib.load(CACHE)
                if isinstance(cached, dict) and cached.get("version") == MODEL_VERSION and "selection" in cached and "faktor" in cached:
                    _B = cached
                else:
                    _B = train()
                    joblib.dump(_B, CACHE, compress=3)
            except Exception:
                _B = train()
                joblib.dump(_B, CACHE, compress=3)
        else:
            _B = train()
            joblib.dump(_B, CACHE, compress=3)
    return _B


def _load_operational_model(B):
    """Muat model operasional terpisah; artefak usang/tidak cocok tidak digunakan."""
    if not os.path.exists(OP_CACHE):
        return None
    try:
        art = joblib.load(OP_CACHE)
        if (art.get("research_version") != MODEL_VERSION
                or art.get("config_key") != B.get("selected_config", {}).get("key")
                or art.get("features") != B.get("features")
                or "model" not in art):
            return None
        return art
    except Exception:
        return None


def _forecast_bundle(B):
    """Gabungkan model operasional yang tervalidasi tanpa mengubah hasil riset/cache evaluasi."""
    art = _load_operational_model(B)
    if not art:
        result = dict(B)
        result["model_source"] = "model penelitian"
        result["model_updated_at"] = None
        return result
    result = dict(B)
    result["model"] = art["model"]
    result["model_source"] = "model operasional tervalidasi"
    result["model_updated_at"] = art.get("created_at")
    result["model_trained_through"] = art.get("trained_through")
    result["operational_validation"] = art.get("validation", {})
    return result


def _validate_and_fit_operational(feat, B):
    """Validasi temporal terhadap baseline sebelum menyimpan model operasional terpisah."""
    f = feat.copy().sort_values(["varian", "tanggal"]).reset_index(drop=True)
    f["tanggal"] = pd.to_datetime(f["tanggal"], errors="coerce").dt.normalize()
    f["gram_terjual"] = pd.to_numeric(f["gram_terjual"], errors="coerce")
    if "label_valid" not in f:
        f["label_valid"] = f["gram_terjual"].notna()
    else:
        f["label_valid"] = f["label_valid"].fillna(False).astype(bool) & f["gram_terjual"].notna()
    if "tanggal_tidak_lengkap" in f:
        f["label_valid"] &= ~f["tanggal_tidak_lengkap"].fillna(True).astype(bool)
    if "gram_feature" not in f:
        f["gram_feature"] = f["gram_terjual"].where(f["label_valid"])
    else:
        f["gram_feature"] = pd.to_numeric(f["gram_feature"], errors="coerce").where(f["label_valid"])
    f["layak_riwayat"] = f.groupby("varian", sort=False).cumcount().ge(28)
    valid_dates = pd.DatetimeIndex(f.loc[f["label_valid"], "tanggal"].dropna().unique()).sort_values()
    if len(valid_dates) < 90:
        raise ValueError(f"Pembaruan model memerlukan sedikitnya 90 tanggal kalender untuk validasi temporal; hanya {len(valid_dates)} tanggal dengan data target valid.")
    last_date = pd.Timestamp(f.loc[f["label_valid"], "tanggal"].max())
    validation_start = last_date - pd.Timedelta(days=27)
    if (validation_start - f["tanggal"].min()).days < 60:
        raise ValueError("Riwayat sebelum periode validasi harus sekurang-kurangnya 60 hari.")
    cfg = B.get("selected_config")
    if not cfg:
        raise ValueError("Konfigurasi model penelitian belum tersedia.")
    validation, _ = _backtest(f, cfg, validation_start, last_date, capture_models=False)
    # Baseline bergulir dan RF dihitung pada pasangan tanggal-varian yang sama.
    base_features = f[["tanggal", "varian", "rm_7"]].copy()
    compare = validation.merge(base_features, on=["tanggal", "varian"], how="left", validate="one_to_one")
    compare = compare.dropna(subset=["gram_terjual", "prediksi", "rm_7"])
    if len(compare) < 30 or compare["tanggal"].nunique() < 7:
        raise ValueError("Data validasi operasional terlalu sedikit setelah target tak lengkap dan baseline disamakan.")
    rf_metrics = metrik(compare["gram_terjual"], compare["prediksi"])
    naive_metrics = metrik(compare["gram_terjual"], compare["rm_7"])
    if rf_metrics["WAPE"] is None or naive_metrics["WAPE"] is None:
        raise ValueError("WAPE tidak dapat dihitung untuk validasi operasional; periksa target aktual.")
    passed = rf_metrics["WAPE"] <= naive_metrics["WAPE"]
    summary = {
        "validation_start": str(validation_start.date()), "validation_end": str(last_date.date()),
        "n_rows": int(len(compare)), "n_days": int(compare["tanggal"].nunique()),
        "rf": rf_metrics, "baseline_rm7": naive_metrics,
        "accepted": bool(passed),
        "rule": "Model hanya diaktifkan jika WAPE validasi temporal tidak lebih buruk daripada baseline rolling mean 7 hari.",
    }
    if not passed:
        return summary
    train_rows = f[ f["label_valid"] & f["layak_riwayat"] & f["gram_terjual"].notna() ].copy()
    model, cap = _fit_model(train_rows, cfg)
    artifact = {
        "research_version": MODEL_VERSION, "config_key": cfg["key"], "features": list(cfg["features"]),
        "model": model, "cap": cap, "validation": summary,
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "trained_through": str(last_date.date()), "n_train_rows": int(len(train_rows)),
    }
    fd, tmp = tempfile.mkstemp(prefix="operational-model-", suffix=".joblib", dir=BASE)
    os.close(fd)
    try:
        joblib.dump(artifact, tmp, compress=3)
        os.replace(tmp, OP_CACHE)
    finally:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except OSError:
            pass
    summary["trained_through"] = artifact["trained_through"]
    summary["n_train_rows"] = artifact["n_train_rows"]
    return summary


# ---------- praproses data mentah ----------
def baca_tanggal(s):
    """Tanggal tanpa ambigu: teks ISO (2025-12-01 00:38) dibaca tahun-bulan-hari; selain itu format lokal hari-bulan-tahun."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    t = s.astype(str).str.strip(); iso = t.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    a = pd.to_datetime(t.where(iso).str.extract(r"^(\d{4}-\d{1,2}-\d{1,2})")[0], format="%Y-%m-%d", errors="coerce")
    b = pd.to_datetime(t.where(~iso), format="%d/%m/%Y %H:%M:%S", errors="coerce")                 # format TikTok-Tokopedia
    b = b.fillna(pd.to_datetime(t.where(~iso & b.isna()), dayfirst=True, errors="coerce"))        # format lokal lain
    return a.fillna(b).dt.normalize()


def ukuran_gram(s, sku=False):
    """Ukuran kemasan (gram) dari teks: '200 gr', '1kg', dst. Untuk SKU juga membaca angka tanpa satuan (LUWP-200, LUWP200). NaN bila tidak ada."""
    s = s.fillna("").astype(str)
    e = s.str.extract(SIZE, flags=re.I)
    g = pd.to_numeric(e[0].str.replace(",", ".", regex=False), errors="coerce") * np.where(e[1].str.lower().isin(["kg", "kilo"]), 1000, 1)
    g = g.where((g >= 20) & (g <= 5000))                           # di luar rentang wajar bukan ukuran kemasan
    if sku:
        ang = pd.to_numeric(s.str.findall(r"(?<!\d)(\d{3,4})(?!\d)").map(lambda L: next((int(x) for x in reversed(L) if int(x) in UKURAN_UMUM), np.nan)), errors="coerce")
        g = g.fillna(ang)
    return g


def standar(raw):
    """Samakan kolom satu file mentah ke kolom baku. Kanal dikenali dari nama kolom: Shopee, TikTok-Tokopedia, atau umum."""
    cols = {norm(c): c for c in raw.columns}
    m = {k: next((cols[norm(a)] for a in al if norm(a) in cols), None) for k, al in ALIAS.items()}
    m = {k: v for k, v in m.items() if v is not None}
    if not {"tanggal", "varian"} <= set(m) or not ({"unit", "gram"} & set(m)):
        raise ValueError("Kolom wajib tidak ditemukan. Dibutuhkan: tanggal, varian/produk, dan jumlah atau gram. Kolom terbaca: " + ", ".join(map(str, raw.columns)))
    df = raw[list(m.values())].copy()
    df.columns = list(m)
    df["kanal"] = "Shopee" if "waktu pesanan dibuat" in cols else "TikTok-Tokopedia" if "seller sku" in cols and "created time" in cols else "Umum"
    return df, m


def praproses(raw, B):
    log, warn = [], []
    parts = [standar(fr) for fr in (raw if isinstance(raw, (list, tuple)) else [raw])]
    for kn in dict.fromkeys(p[0]["kanal"].iat[0] for p in parts if len(p[0])):
        ps = [p for p in parts if len(p[0]) and p[0]["kanal"].iat[0] == kn]
        log.append(f"{kn}: {len(ps):,} file, {sum(len(p[0]) for p in ps):,} baris dibaca. Kolom dipetakan: " + ", ".join(f"{k} ← {v}" for k, v in ps[0][1].items()))
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    n = len(df); df = df.drop_duplicates()
    log.append(f"{n - len(df):,} baris duplikat persis dihapus.")
    if "status" in df:
        bad = df["status"].fillna("").astype(str).str.contains("batal|cancel|gagal|refund|retur|return", case=False, regex=True)
        log.append(f"{int(bad.sum()):,} pesanan batal/retur dikeluarkan."); df = df[~bad]
    if "jenis" in df:
        cn = df["jenis"].fillna("").astype(str).str.contains("cancel|batal", case=False, regex=True)
        log.append(f"{int(cn.sum()):,} baris berjenis Cancel dikeluarkan (Return/Refund ditangani lewat kuantitas retur bila tersedia)."); df = df[~cn]
    df["tanggal"] = baca_tanggal(df["tanggal"])
    df["varian"] = df["varian"].fillna("").astype(str).str.strip()
    ok = df["tanggal"].notna() if "sku" in df else df["tanggal"].notna() & ~df["varian"].str.lower().isin(KOSONG)
    log.append(f"{int((~ok).sum()):,} baris tanpa tanggal valid" + ("" if "sku" in df else " atau nama varian") + " dibuang.")
    df = df[ok].copy()
    if df.empty:
        raise ValueError("Tidak ada baris valid setelah pembersihan. Periksa format tanggal dan nama varian.")
    if df["kanal"].nunique() > 1:
        akhir = df.groupby("kanal")["tanggal"].max(); batas = akhir.min()
        if (akhir > batas).any():
            log.append("Tanggal terakhir tiap kanal: " + ", ".join(f"{k} {v.date()}" for k, v in akhir.items()) + f". Data dipotong sampai {batas.date()} ({int((df['tanggal'] > batas).sum()):,} baris) agar semua kanal memiliki batas waktu yang sama.")
            df = df[df["tanggal"] <= batas].copy()
    suspect_dates = set()
    for kn, g in df.groupby("kanal"):
        counts = g.groupby("tanggal").size()
        counts = counts.reindex(pd.date_range(counts.index.min(), counts.index.max(), freq="D"), fill_value=0)
        suspicious = counts[(counts == 0) & (counts.rolling(15, center=True, min_periods=8).median() >= 10)]
        if len(suspicious):
            suspect_dates.update(suspicious.index.normalize())
            warn.append(f"{kn}: {len(suspicious)} tanggal tidak memiliki baris pesanan padahal hari sekitarnya ramai. Tanggal tersebut ditandai data tidak lengkap, bukan dianggap penjualan nol.")
    unit = pd.to_numeric(df["unit"], errors="coerce") if "unit" in df else None
    if unit is not None and "retur" in df:
        rt = pd.to_numeric(df["retur"], errors="coerce").fillna(0).clip(lower=0)
        if (rt > 0).any():
            log.append(f"{int((rt > 0).sum()):,} baris punya retur tercatat; {rt.sum():,.0f} unit dikurangkan dari penjualan.")
        unit = (unit - rt).clip(lower=0)
    if "sku" in df:
        sku_t = df["sku"].fillna("").astype(str).str.upper().str.strip()
        judul = sku_t.str.contains(r"\s") & (sku_t.str.len() > 20)
        kd = sku_t.str.extract(r"^([A-Z]+)")[0].where(~judul)
        peta = {c.strip().upper(): v for k, v in B["sku"].items() for c in str(k).split(",")}
        vr = kd.map(peta); luar = vr.isna()
        if luar.any():
            uk = ukuran_gram(df["sku"], sku=True)
            if "variasi" in df: uk = uk.fillna(ukuran_gram(df["variasi"]))
            gbx = uk * (unit if unit is not None else 1)
            lp = pd.DataFrame({"kode": kd.where(kd.notna(), pd.Series(np.where(judul, "(SKU berupa judul produk)", sku_t.replace("", "(SKU kosong)")), index=df.index)), "kg": gbx / 1000})[luar]
            tb = lp.groupby("kode").agg(baris=("kg", "size"), kg=("kg", "sum")).sort_values("kg", ascending=False)
            kg_luar, kg_dalam = float(lp["kg"].sum()), float(gbx[~luar].sum() / 1000)
            log.append(f"{int(luar.sum()):,} baris ({len(tb):,} kode SKU) di luar varian sampel tidak diprediksi. Terbesar: " + "; ".join(f"{k} {r.kg:,.1f} kg" for k, r in tb.head(6).iterrows()) + ".")
            warn.append(f"Prediksi mencakup sekitar {kg_dalam / max(kg_dalam + kg_luar, 1e-9) * 100:.0f}% volume teridentifikasi pada file; {len(tb):,} kode SKU tidak memiliki model/faktor terverifikasi.")
        df["varian"] = vr
        df = df[~luar].copy(); unit = unit[~luar] if unit is not None else None
        if df.empty:
            raise ValueError("Tidak ada kode SKU yang cocok dengan varian acuan model. Periksa sheet Referensi_Varian.")
    gram = pd.to_numeric(df["gram"], errors="coerce") if "gram" in df else pd.Series(np.nan, index=df.index)
    nm = ukuran_gram(df["variasi"]) if "variasi" in df else pd.Series(np.nan, index=df.index)
    if "sku" not in df: nm = ukuran_gram(df["varian"]).fillna(nm)
    sk = ukuran_gram(df["sku"], sku=True) if "sku" in df else pd.Series(np.nan, index=df.index)
    pack = sk.fillna(nm)
    butuh = gram.isna() & unit.notna()
    gram = gram.where(~butuh, unit * pack)
    unresolved = gram.isna() | (gram < 0)
    if unresolved.any():
        warn.append(f"{int(unresolved.sum()):,} baris transaksi dihapus karena jumlah gram tidak dapat ditentukan secara valid; sistem tidak mengubah data tidak diketahui menjadi nol.")
        df = df[~unresolved].copy(); gram = gram[~unresolved]; unit = unit[~unresolved] if unit is not None else None
    if df.empty:
        raise ValueError("Tidak ada transaksi dengan jumlah gram valid. Lengkapi kolom gram atau ukuran SKU.")
    df["gram_terjual"] = gram.clip(lower=0)
    df["terjual_unit"] = (unit if unit is not None else pd.Series(1, index=df.index)).fillna(0).clip(lower=0)

    def canon(s):
        text = re.sub(r"\s+", " ", re.sub(SIZE, "", s, flags=re.I)).strip(" -/|,").upper()
        if text in B["sku"]: return B["sku"][text]
        t = re.sub(r"[-_ /]*(?:100|200|500|1000)$", "", text)
        if t in B["sku"]: return B["sku"][t]
        if text in B["known"]: return text
        match = difflib.get_close_matches(text, B["known"], n=1, cutoff=0.90)
        return match[0] if match else text
    if "sku" not in df:
        mp = {s: canon(str(s)) for s in df["varian"].unique()}
        df["varian"] = df["varian"].map(mp)
    else:
        # SKU yang dikenali sudah dipetakan ke varian di atas.
        df["varian"] = df["varian"].astype(str).str.strip()
    empty_variant = df["varian"].str.strip().str.lower().isin(KOSONG)
    if empty_variant.any():
        df = df[~empty_variant].copy(); warn.append(f"{int(empty_variant.sum()):,} baris varian kosong dikeluarkan.")
    daily = df.groupby(["tanggal", "varian"], as_index=False)[["terjual_unit", "gram_terjual"]].sum()
    last, parts = daily["tanggal"].max(), []
    for v, g in daily.groupby("varian"):
        idx = pd.date_range(g["tanggal"].min(), last, freq="D")
        g = g.set_index("tanggal")[["terjual_unit", "gram_terjual"]].reindex(idx)
        # Hari tanpa transaksi diasumsikan nol hanya jika tidak ditandai sebagai kemungkinan ekspor tidak lengkap.
        g[["terjual_unit", "gram_terjual"]] = g[["terjual_unit", "gram_terjual"]].fillna(0)
        g = g.rename_axis("tanggal").reset_index(); g["varian"] = v; parts.append(g)
    full = pd.concat(parts, ignore_index=True)
    if suspect_dates:
        full.loc[full["tanggal"].isin(suspect_dates), ["terjual_unit", "gram_terjual"]] = np.nan
        latest = full["tanggal"].max()
        if latest in suspect_dates:
            last_complete = full.loc[~full["tanggal"].isin(suspect_dates), "tanggal"].max()
            full = full[full["tanggal"] <= last_complete].copy()
            log.append(f"Tanggal akhir dipotong ke {last_complete.date()} karena ekspor pada tanggal yang lebih baru terindikasi tidak lengkap.")
    full["gram_feature"] = full["gram_terjual"]
    full["label_valid"] = full["gram_terjual"].notna()
    full["tanggal_tidak_lengkap"] = full["gram_terjual"].isna()
    full["fase"] = "data baru"
    full = add_features(full, target_col="gram_feature")
    log.append(f"Deret harian dibuat untuk {full['varian'].nunique()} varian; hari tanpa transaksi dianggap nol hanya pada tanggal yang tidak ditandai mencurigakan.")
    above = int((full["gram_terjual"] > B["cap"]).sum())
    log.append(f"{above:,} observasi melampaui persentil 99,5 pada data latih ({B['cap']:,.0f} g/hari). Lonjakan dipertahankan; tidak dipangkas secara default.")
    days = full.groupby("varian")["tanggal"].nunique()
    short = days[days < 28]
    if len(short):
        warn.append(f"Riwayat kurang dari 28 hari pada {len(short)} varian. Unggah minimal 28 hari, idealnya 60 hari atau lebih.")
    matched = set(full["varian"]) & set(B["faktor"])
    missing_factor = sorted(set(full["varian"]) - set(B["faktor"]))
    if missing_factor:
        warn.append("Rate penyusutan belum tersedia untuk: " + ", ".join(missing_factor[:10]) + ". Estimasi green bean hanya dihitung untuk varian yang mempunyai rate.")
    provisional = [v for v in matched if not str(B.get("pemetaan_status", {}).get(v, "")).lower().startswith("cocok")]
    if provisional:
        warn.append(f"Estimasi green bean tetap dihitung menggunakan rate pada Stock.xlsx untuk {len(provisional)} varian yang nama produk/SKU-nya masih perlu konfirmasi perusahaan. Periksa status pemetaan pada tabel sebelum memakai estimasi sebagai acuan produksi atau pengadaan.")
    return full, log, warn


# ---------- output prediksi kebutuhan bahan baku ----------
# Kolom utama forecasting tidak mensyaratkan snapshot stok gudang. Kolom inventori
# tetap dipertahankan di fungsi kebutuhan() untuk pengembangan perencanaan neto terpisah.
FORECAST_OUTPUT_COLUMNS = [
    "varian", "roasted_g", "roasted_kg", "penyusutan_pct", "faktor", "green_g", "green_kg",
    "status_pemetaan", "nama_penyusutan_sumber", "akurasi", "galat_kg", "galat_green_kg",
    "buffer_pct", "permintaan_terlindungi_kg", "green_terlindungi_kg"
]


# ---------- perencanaan stok neto (opsional/terpisah dari forecasting) ----------
STOCK_FIELDS = [
    "stok_produk_jadi_kg", "pasokan_produk_jadi_kg", "target_akhir_produk_jadi_kg",
    "stok_roasted_kg", "pasokan_roasted_kg", "target_akhir_roasted_kg",
    "stok_green_kg", "pasokan_green_kg", "target_akhir_green_kg",
]
STOCK_ALIASES = {
    "varian": ["varian", "nama varian", "varian acuan", "produk"],
    "stok_produk_jadi_kg": ["stok_produk_jadi_kg", "stok produk jadi kg", "stok produk jadi (kg)"],
    "pasokan_produk_jadi_kg": ["pasokan_produk_jadi_kg", "produk jadi dipesan kg", "pasokan produk jadi kg"],
    "target_akhir_produk_jadi_kg": ["target_akhir_produk_jadi_kg", "target stok akhir produk jadi kg", "target akhir produk jadi kg"],
    "stok_roasted_kg": ["stok_roasted_kg", "stok roasted kg", "stok roasted bean kg"],
    "pasokan_roasted_kg": ["pasokan_roasted_kg", "pasokan roasted kg", "roasted dipesan kg"],
    "target_akhir_roasted_kg": ["target_akhir_roasted_kg", "target stok akhir roasted kg", "target akhir roasted kg"],
    "stok_green_kg": ["stok_green_kg", "stok green bean kg", "stok green kg"],
    "pasokan_green_kg": ["pasokan_green_kg", "pasokan green bean kg", "green dipesan kg"],
    "target_akhir_green_kg": ["target_akhir_green_kg", "target stok akhir green bean kg", "target akhir green kg"],
}


def baca_file_stok(upload):
    if upload is None or not getattr(upload, "filename", ""):
        return None, []
    filename = upload.filename.lower()
    if filename.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(upload)
    elif filename.endswith(".csv"):
        payload = upload.read(); raw = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                raw = pd.read_csv(io.BytesIO(payload), sep=None, engine="python", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if raw is None:
            raise ValueError("File stok CSV tidak dapat dibaca. Simpan sebagai CSV UTF-8.")
    else:
        raise ValueError("File stok harus CSV atau XLSX.")
    cols = {norm(c): c for c in raw.columns}
    mapping = {k: next((cols[norm(a)] for a in aliases if norm(a) in cols), None) for k, aliases in STOCK_ALIASES.items()}
    if mapping["varian"] is None:
        raise ValueError("Kolom wajib file stok adalah 'varian'. Unduh template stok dari halaman prediksi.")
    inv = pd.DataFrame({"varian": raw[mapping["varian"]].astype(str).str.strip()})
    warnings = []
    for field in STOCK_FIELDS:
        col = mapping[field]
        if col is None:
            inv[field] = np.nan
            warnings.append(f"Kolom {field} belum tersedia; rekomendasi terkait tidak dihitung.")
            continue
        def parse_num(v):
            if pd.isna(v) or str(v).strip() == "": return np.nan
            if isinstance(v, (int, float, np.number)): return float(v)
            txt = str(v).strip().replace(" ", "")
            if "," in txt and "." in txt:
                if txt.rfind(",") > txt.rfind("."):
                    txt = txt.replace(".", "").replace(",", ".")
                else:
                    txt = txt.replace(",", "")
            elif "," in txt:
                txt = txt.replace(",", ".")
            return pd.to_numeric(txt, errors="coerce")
        inv[field] = raw[col].map(parse_num)
        negatives = inv[field] < 0
        if negatives.any():
            raise ValueError(f"Kolom {field} memiliki {int(negatives.sum())} nilai negatif. Persediaan/pasokan harus nol atau positif.")
    inv["varian_key"] = inv["varian"].astype(str).str.strip().str.upper()
    if inv["varian_key"].duplicated().any():
        dups = inv.loc[inv["varian_key"].duplicated(False), "varian"].unique().tolist()
        raise ValueError("File stok memiliki varian ganda: " + ", ".join(dups[:6]) + ". Gunakan satu baris per varian.")
    return inv, warnings


def statistik(B, hz, tl):
    # Buffer/indikator kualitas operasional dikalibrasi hanya dengan forecast
    # rekursif periode validasi. Backtest test tetap dipakai terpisah untuk laporan
    # penelitian dan tidak memengaruhi tambahan kebutuhan otomatis.
    k = 7 if hz <= 7 else 14 if hz <= 14 else 30
    calibration = B.get("buffer_calibration", B["rek"])
    r = calibration[calibration["hz"] == k]
    st = {v: metrik_rek(g, tl) for v, g in r.groupby("varian")} if not r.empty else {}
    if not r.empty:
        total = r.groupby("o", as_index=False)[["gram_terjual", "prediksi"]].sum()
        st["__total__"] = metrik_rek(total, tl)
    else:
        st["__total__"] = {"akurasi": None, "galat_kg": None, "bias": None, "buffer": np.nan}
    st["__hz__"] = k
    st["__buffer_source__"] = "validasi" if "buffer_calibration" in B else "fallback-backtest"
    st["__origins__"] = int(r["o"].nunique()) if not r.empty and "o" in r else 0
    return st


def kebutuhan(fc, B, buf, st, inventory=None):
    t = fc.groupby("varian", as_index=False)["prediksi"].sum()
    t["roasted_g"] = pd.to_numeric(t.pop("prediksi"), errors="coerce").clip(lower=0)
    t["roasted_kg"] = t["roasted_g"] / 1000.0
    t["penyusutan_pct"] = t["varian"].map(B.get("penyusutan", {})) * 100.0
    t["faktor_default"] = ~t["varian"].isin(B["faktor"])
    t["faktor"] = t["varian"].map(B["faktor"])
    t["green_g"] = t["roasted_g"] * t["faktor"]
    t["green_kg"] = t["green_g"] / 1000.0
    t["status_pemetaan"] = t["varian"].map(B.get("pemetaan_status", {})).fillna("Rate penyusutan belum tersedia")
    t["nama_penyusutan_sumber"] = t["varian"].map(B.get("nama_penyusutan_sumber", {})).fillna("")
    t["akurasi"] = t["varian"].map(lambda v: st.get(v, {}).get("akurasi", st.get("__total__", {}).get("akurasi")))
    t["galat_kg"] = t["varian"].map(lambda v: st.get(v, {}).get("galat_kg", st.get("__total__", {}).get("galat_kg")))
    if buf is not None:
        t["buffer_pct"] = float(buf)
    else:
        t["buffer_pct"] = t["varian"].map(lambda v: st.get(v, {}).get("buffer", st.get("__total__", {}).get("buffer", np.nan)))
    t["buffer_pct"] = pd.to_numeric(t["buffer_pct"], errors="coerce").clip(lower=0)
    # Estimasi bruto berdasarkan forecast; tidak perlu snapshot stok untuk menghitungnya.
    t["permintaan_terlindungi_kg"] = t["roasted_kg"] * (1 + t["buffer_pct"] / 100)
    t["green_terlindungi_kg"] = t["permintaan_terlindungi_kg"] * t["faktor"]
    t["galat_green_kg"] = t["galat_kg"] * t["faktor"]
    for field in STOCK_FIELDS:
        t[field] = np.nan
    t["stok_tersedia"] = False
    t["produksi_produk_jadi_kg"] = np.nan
    t["roasted_perlu_diproduksi_kg"] = np.nan
    t["green_perlu_bruto_kg"] = np.nan
    t["rekomendasi_green_beli_kg"] = np.nan
    t["rekomendasi_roasted_beli_kg"] = np.nan
    if inventory is not None:
        inv = inventory.copy().set_index("varian_key")
        t["varian_key"] = t["varian"].astype(str).str.strip().str.upper()
        for field in STOCK_FIELDS:
            t[field] = t["varian_key"].map(inv[field])
        missing_var = ~t["varian_key"].isin(inv.index)
        required_ok = t[STOCK_FIELDS].notna().all(axis=1) & ~missing_var
        t["stok_tersedia"] = required_ok
        t.loc[required_ok, "produksi_produk_jadi_kg"] = (
            t.loc[required_ok, "permintaan_terlindungi_kg"] + t.loc[required_ok, "target_akhir_produk_jadi_kg"]
            - t.loc[required_ok, "stok_produk_jadi_kg"] - t.loc[required_ok, "pasokan_produk_jadi_kg"]).clip(lower=0)
        t.loc[required_ok, "roasted_perlu_diproduksi_kg"] = (
            t.loc[required_ok, "produksi_produk_jadi_kg"] + t.loc[required_ok, "target_akhir_roasted_kg"]
            - t.loc[required_ok, "stok_roasted_kg"] - t.loc[required_ok, "pasokan_roasted_kg"]).clip(lower=0)
        factor_ok = required_ok & t["faktor"].notna()
        t.loc[factor_ok, "green_perlu_bruto_kg"] = t.loc[factor_ok, "roasted_perlu_diproduksi_kg"] * t.loc[factor_ok, "faktor"]
        t.loc[factor_ok, "rekomendasi_green_beli_kg"] = (
            t.loc[factor_ok, "green_perlu_bruto_kg"] + t.loc[factor_ok, "target_akhir_green_kg"]
            - t.loc[factor_ok, "stok_green_kg"] - t.loc[factor_ok, "pasokan_green_kg"]).clip(lower=0)
        t.loc[required_ok, "rekomendasi_roasted_beli_kg"] = t.loc[required_ok, "roasted_perlu_diproduksi_kg"]
        t = t.drop(columns=["varian_key"])
    t["roasted_stok_kg"] = t["roasted_perlu_diproduksi_kg"]
    t["green_stok_kg"] = t["rekomendasi_green_beli_kg"]
    return t.sort_values("roasted_kg", ascending=False)[[
        "varian", "roasted_g", "roasted_kg", "penyusutan_pct", "faktor", "green_g", "green_kg",
        "status_pemetaan", "nama_penyusutan_sumber", "akurasi", "galat_kg", "galat_green_kg", "buffer_pct",
        "permintaan_terlindungi_kg", "green_terlindungi_kg", "stok_produk_jadi_kg", "pasokan_produk_jadi_kg", "target_akhir_produk_jadi_kg",
        "produksi_produk_jadi_kg", "stok_roasted_kg", "pasokan_roasted_kg", "target_akhir_roasted_kg",
        "roasted_perlu_diproduksi_kg", "rekomendasi_roasted_beli_kg", "stok_green_kg", "pasokan_green_kg",
        "target_akhir_green_kg", "green_perlu_bruto_kg", "rekomendasi_green_beli_kg", "stok_tersedia", "faktor_default"
    ]]


# ---------- analisis AI: Claude bila ada kunci API, jika tidak aturan otomatis (tanpa internet) ----------
def idr(x, d=1):
    try:
        if x is None or not np.isfinite(float(x)):
            return "belum dihitung"
        return f"{float(x):,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    except (TypeError, ValueError):
        return "belum dihitung"


def analisis_aturan(c):
    r, T = c["ringkas"], c["tabel"]
    cara = "diisi manual" if r["mode"] == "manual" else f"dikalibrasi dari galat ramalan rekursif pada data validasi dengan tingkat {r['tingkat']}%"
    green_line = f" ekuivalen green bean sekitar {idr(r['green'])} kg berdasarkan rate penyusutan per varian" if r.get("green") is not None else "; total green bean belum lengkap karena ada varian tanpa rate penyusutan"
    out = [f"Kesimpulan: pada periode {r['prediksi']} ({c['hz']} hari), prediksi kebutuhan roasted bean adalah {idr(r['roasted'])} kg{green_line}. "
           f"Pada validasi ramalan {r['dasar_hz']} hari ke depan, akurasi ringkas historis {idr(r['akurasi'])}% dan MAE sekitar {idr(r['galat_roasted'])} kg per varian-periode. Angka ini membantu membaca performa, bukan jaminan hasil mendatang."]
    out.append(f"Kebutuhan bahan baku dihitung langsung dari hasil prediksi tanpa memerlukan data stok gudang: estimasi roasted {idr(r.get('roasted'))} kg dan green bean {idr(r.get('green'))} kg. Dengan buffer prediksi rata-rata {idr(r.get('buffer_rata'))}% ({cara}), kebutuhan menjadi sekitar {idr(r.get('roasted_buffer'))} kg roasted dan {idr(r.get('green_buffer'))} kg green bean. Angka ini adalah kebutuhan bruto, bukan jumlah pembelian neto setelah dikurangi persediaan yang ada.")
    tr = c.get("tren", 0)
    out.append("Tren: rata-rata penjualan harian diperkirakan " + (f"relatif stabil ({idr(tr)}% dibanding 28 hari terakhir)." if abs(tr) < 3 else
               f"naik sekitar {idr(tr)}% dibanding 28 hari terakhir." if tr > 0 else
               f"turun sekitar {idr(-tr)}% dibanding 28 hari terakhir."))
    rendah = [t for t in T if t.get("akurasi") is not None and t["akurasi"] < 75][:3]
    if rendah:
        out.append("Varian yang perlu dipantau: " + "; ".join(f"{t['varian']} (akurasi {idr(t['akurasi'])}%)" for t in rendah) + ". Periksa penjualan dan promosi terbaru sebelum membuat keputusan stok.")
    kurang = [t["varian"] for t in T if (t.get("bias") or 0) > 5][:3]
    if kurang:
        out.append("Pada " + ", ".join(kurang) + ", model cenderung memprediksi di bawah penjualan aktual pada backtest; tinjau buffer dan data terbaru.")
    if c.get("warn"):
        out.append("Keterbatasan data: " + " ".join(c["warn"][:3]))
    if c["hz"] > r["dasar_hz"]:
        out.append(f"Catatan akurasi: periode {c['hz']} hari lebih panjang dari horizon terjauh yang diuji ({r['dasar_hz']} hari); akurasi horizon panjang belum terukur langsung.")
    out.append("Fitur tanggal kembar, gajian, dan akhir tahun adalah proksi kalender. Tanpa kalender promosi aktual, sistem tidak dapat mengklaim bahwa lonjakan disebabkan oleh promosi.")
    return out


GEMINI_BASE = os.environ.get("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta")
SISTEM_ANALISIS = ("Anda analis forecasting kebutuhan bahan baku kopi untuk PT Semekar Kreatif Nusantara. Dari data JSON hasil model Random Forest, tulis analisis berbahasa Indonesia yang mudah dipahami, "
                   "tanpa markdown dan tanpa tanda panah. Fokus: periode prediksi, total kebutuhan roasted berdasarkan ramalan penjualan, konversi green bean dari tingkat penyusutan per varian, kebutuhan dengan buffer bila tersedia, varian yang perlu diwaspadai, serta keterbatasan pemetaan penyusutan. "
                   "Tidak perlu meminta data stok untuk menghitung kebutuhan bruto. Tegaskan bahwa angka kebutuhan bruto berbeda dari jumlah pembelian neto setelah dikurangi stok gudang. "
                   "Pakai hanya angka yang ada pada data, jangan mengarang atau memperkirakan angka baru. Tulis 4 sampai 6 paragraf pendek, tiap paragraf diawali label seperti 'Kesimpulan:'.")
SISTEM_BANDING = ("Anda asisten penelitian untuk skripsi prediksi penjualan kopi memakai Random Forest. Dari data JSON hasil perbandingan metode pada data uji yang sama, tulis kesimpulan berbahasa Indonesia "
                  "yang jujur dan mudah dijelaskan saat sidang, tanpa markdown dan tanpa tanda panah. Isi: apakah Random Forest lebih baik daripada metode pembanding pada tiap tingkat (harian, 7 hari, 30 hari), "
                  "pada ukuran apa unggul dan pada ukuran apa tidak, keterbatasan jumlah titik uji, dan cara menyampaikannya secara jujur. Pakai hanya angka pada data, jangan mengarang. "
                  "Tulis 4 sampai 5 paragraf pendek, tiap paragraf diawali label seperti 'Kesimpulan:'.")


def _galat_http(e):
    try:
        pesan = json.load(e)["error"]["message"]
    except Exception:
        pesan = str(getattr(e, "reason", ""))
    return f"HTTP {e.code}: {str(pesan)[:240]}"


def bersih_teks(t):
    """Hapus penanda markdown dari jawaban model; kembalikan daftar paragraf."""
    t = re.sub(r"(\*\*|__|`)", "", t)
    return [re.sub(r"^\s*(#{1,6}\s*|[*-]\s+)", "", p).strip() for p in t.split("\n") if p.strip()]


def panggil_gemini(sistem, isi):
    """Panggil Gemini (Google AI Studio). Kunci dari GEMINI_API_KEY, model dari GEMINI_MODEL (bawaan gemini-flash-latest, cadangan gemini-2.5-flash)."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def kirim(model, tipis):
        gc = {"temperature": 0.2, "maxOutputTokens": 2500}
        if tipis: gc["thinkingConfig"] = {"thinkingBudget": 0}                 # jawaban singkat: matikan "berpikir" agar token tidak habis sebelum teks keluar
        body = {"systemInstruction": {"parts": [{"text": sistem}]}, "contents": [{"role": "user", "parts": [{"text": isi}]}], "generationConfig": gc}
        req = urllib.request.Request(f"{GEMINI_BASE}/models/{model}:generateContent", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        t = "".join(p.get("text", "") for p in d["candidates"][0]["content"].get("parts", [])).strip()
        if not t: raise RuntimeError("jawaban kosong (diblokir atau terpotong)")
        return t

    galat = []
    for model in dict.fromkeys([os.environ.get("GEMINI_MODEL", "gemini-flash-latest"), "gemini-2.5-flash", "gemini-flash-lite-latest"]):
        for tipis in (True, False):
            try:
                return kirim(model, tipis), model
            except urllib.error.HTTPError as e:
                g = _galat_http(e); galat.append(f"{model}: {g}")
                if e.code in (401, 403) or "api key" in g.lower():
                    raise RuntimeError(f"Kunci API ditolak ({g}). Periksa GEMINI_API_KEY; kunci baru AI Studio berawalan AQ. dan dikirim lewat header x-goog-api-key. Bila tetap ditolak, buat kunci baru di Google AI Studio.")
                if e.code == 429:
                    raise RuntimeError(f"Kuota gratis atau batas permintaan per menit tercapai ({g}). Coba lagi beberapa saat lagi.")
                if e.code == 400 and tipis: continue                          # model tanpa pengaturan thinking: ulangi tanpa itu
                break                                                         # model tidak ada atau ditolak: coba model berikutnya
            except urllib.error.URLError as e:
                raise RuntimeError(f"Tidak bisa menghubungi Gemini, periksa koneksi internet ({e.reason}).")
    raise RuntimeError(" | ".join(galat))


def panggil_claude(sistem, isi):
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", method="POST",
                                 data=json.dumps({"model": model, "max_tokens": 1200, "system": sistem, "messages": [{"role": "user", "content": isi}]}).encode(),
                                 headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return "".join(b.get("text", "") for b in json.load(r)["content"]).strip(), model


def panggil_llm(sistem, isi):
    """Pilih penyedia AI dari kunci yang tersedia (Gemini lebih dulu, lalu Claude). Hasil: (paragraf, penyedia, model) atau None bila tanpa kunci."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        t, m = panggil_gemini(sistem, isi); return bersih_teks(t), "gemini", m
    if os.environ.get("ANTHROPIC_API_KEY"):
        t, m = panggil_claude(sistem, isi); return bersih_teks(t), "claude", m
    return None


def jawab_ai(sistem, data, cadangan):
    """Jalankan AI bila ada kunci; bila gagal atau tanpa kunci, pakai teks aturan otomatis dan sertakan alasannya."""
    try:
        r = panggil_llm(sistem, json.dumps(data, ensure_ascii=False))
        if r: return {"sumber": r[1], "model": r[2], "teks": r[0]}
        return {"sumber": "aturan", "teks": cadangan(), "galat": "Kunci AI belum diisi (GEMINI_API_KEY)."}
    except Exception as e:
        print("Analisis AI gagal, memakai aturan otomatis:", e)
        return {"sumber": "aturan", "teks": cadangan(), "galat": str(e)[:500]}


def analisis_ai(c):
    data = {"periode": c["ringkas"]["prediksi"], "hari": c["hz"], "ringkasan": c["ringkas"], "tren_harian_persen": round(c["tren"], 1), "varian_teratas": c["tabel"], "peringatan": c["warn"]}
    return jawab_ai(SISTEM_ANALISIS, data, lambda: analisis_aturan(c))


# ---------- aplikasi web ----------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ganti-kunci-rahasia-ini")
USERS = {os.environ.get("APP_USER", "admin"): generate_password_hash(os.environ.get("APP_PASS", "semekar2026"))}
LAST = {}


def login_required(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        return fn(*a, **k) if "u" in session else redirect(url_for("login"))
    return w


@app.route("/login", methods=["GET", "POST"])
def login():
    err = None
    if request.method == "POST":
        u, p = request.form.get("username", "").strip(), request.form.get("password", "")
        if u in USERS and check_password_hash(USERS[u], p):
            session["u"] = u
            return redirect(url_for("dashboard"))
        err = "Nama pengguna atau kata sandi salah."
    return render_template("login.html", err=err)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", page="dashboard")


@app.route("/prediksi")
@login_required
def prediksi():
    return render_template("predict.html", page="prediksi")


@app.route("/api/analisis")
@login_required
def api_analisis():
    B = bundle(); bt = B["bt"]
    rows = [[t.strftime("%Y-%m-%d"), v, float(a), round(float(p))]
            for t, v, a, p in zip(bt["tanggal"], bt["varian"], bt["gram_terjual"], bt["prediksi"])]
    return jsonify(kpi=B["metrik"], split=B["split"], imp=B["imp"], stok=B["stok"], pv=B["pv"],
                   lv=tabel_level(B), rows=rows, selection=B["selection"], audit=B["audit"])


@app.route("/api/latih-ulang", methods=["POST"])
@login_required
def latih_ulang():
    try:
        bundle(force=True)
        return jsonify(ok=True)
    except Exception as e:
        app.logger.exception("Gagal melatih ulang model")
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/latih-operasional", methods=["POST"])
@login_required
def latih_operasional():
    """Perbarui model operasional dari ekspor transaksi setelah lolos validasi temporal."""
    try:
        files = [u for u in request.files.getlist("file") if u and u.filename]
        if not files:
            raise ValueError("Pilih file transaksi CSV/XLSX yang mencakup riwayat terbaru sebelum memperbarui model.")
        frames = []
        for u in files:
            filename = u.filename.lower()
            if filename.endswith((".xlsx", ".xls")):
                fr = pd.read_excel(u)
            elif filename.endswith(".csv"):
                payload = u.read(); fr = None
                for enc in ("utf-8-sig", "utf-8", "latin-1"):
                    try:
                        fr = pd.read_csv(io.BytesIO(payload), sep=None, engine="python", encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if fr is None:
                    raise ValueError(f"File {u.filename} tidak bisa dibaca sebagai CSV. Simpan sebagai CSV UTF-8.")
            else:
                raise ValueError(f"Format file {u.filename} tidak didukung. Gunakan CSV atau XLSX.")
            frames.append(fr)
        B = bundle()
        feat, log, warn = praproses(frames, B)
        result = _validate_and_fit_operational(feat, B)
        result["source_files"] = [u.filename for u in files]
        result["warnings"] = warn
        if not result["accepted"]:
            result["message"] = "Pembaruan ditolak: WAPE model Random Forest pada validasi terbaru lebih buruk daripada baseline rata-rata 7 hari. Model operasional sebelumnya tidak diubah."
            return jsonify(result), 200
        result["message"] = "Model operasional berhasil diperbarui setelah lolos validasi temporal. Hasil evaluasi penelitian tidak diubah."
        return jsonify(result), 200
    except Exception as e:
        app.logger.exception("Pembaruan model operasional gagal")
        return jsonify(error=str(e)), 400


@app.route("/unduh/template-stok")
@login_required
def template_stok():
    B = bundle()
    template = pd.DataFrame({"varian": B["known"]})
    for c in STOCK_FIELDS:
        template[c] = ""
    return Response(template.to_csv(index=False), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=template_persediaan_semekar.csv"})


@app.route("/api/prediksi", methods=["POST"])
@login_required
def api_prediksi():
    try:
        files = [u for u in request.files.getlist("file") if u and u.filename]
        if not files:
            raise ValueError("Pilih minimal satu file transaksi CSV atau XLSX terlebih dahulu.")
        frames = []
        for u in files:
            filename = u.filename.lower()
            if filename.endswith((".xlsx", ".xls")):
                fr = pd.read_excel(u)
            elif filename.endswith(".csv"):
                payload = u.read(); fr = None
                for enc in ("utf-8-sig", "utf-8", "latin-1"):
                    try:
                        fr = pd.read_csv(io.BytesIO(payload), sep=None, engine="python", encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if fr is None:
                    raise ValueError(f"File {u.filename} tidak bisa dibaca sebagai CSV. Simpan sebagai CSV UTF-8.")
            else:
                raise ValueError(f"Format file {u.filename} tidak didukung. Gunakan CSV atau XLSX.")
            frames.append(fr)
        B = bundle()
        feat, log, warn = praproses(frames, B)
        forecast_B = _forecast_bundle(B)
        log.append(f"Model prediksi aktif: {forecast_B['model_source']}" + (f" (data pelatihan sampai {forecast_B.get('model_trained_through')}; diperbarui {forecast_B.get('model_updated_at')})." if forecast_B.get('model_source') == 'model operasional tervalidasi' else "."))
        # Forecasting kebutuhan bruto tidak memerlukan file persediaan. Data inventori
        # hanya dibutuhkan untuk fitur terpisah yang menghitung kebutuhan beli neto.
        inventory = None
        if len(files) > 1:
            log.insert(0, f"{len(files)} file transaksi digabung: " + ", ".join(u.filename for u in files[:4]) + (f" dan {len(files) - 4} file lain" if len(files) > 4 else "."))
        if feat["tanggal"].nunique() < 28:
            return jsonify(error=f"Riwayat pada file hanya {feat['tanggal'].nunique()} hari ({feat['tanggal'].min().date()} s.d. {feat['tanggal'].max().date()}); fitur lag memerlukan minimal 28 hari dan idealnya 60 hari. Unggah riwayat lebih panjang.", log=log, warn=warn), 400
        horizon_text = request.form.get("horizon", "30")
        try:
            hz = min(max(int(horizon_text), 7), 90)
        except (TypeError, ValueError):
            raise ValueError("Horizon prediksi harus berupa bilangan hari antara 7 dan 90.")
        b = request.form.get("buffer", "").strip()
        if b:
            try: buf = min(max(float(b.replace(",", ".")), 0), 100)
            except ValueError: raise ValueError("Buffer manual harus berupa angka antara 0 dan 100 persen.")
        else:
            buf = None
        try: tl = min(max(float(request.form.get("tingkat", 90)), 50), 99) / 100
        except (TypeError, ValueError): tl = 0.90
        fc = forecast(feat, forecast_B, hz)
        st = statistik(B, hz, tl)
        tab = kebutuhan(fc, B, buf, st, inventory=inventory)
        if tab["faktor"].isna().any():
            missing_factor = int(tab["faktor"].isna().sum())
            warn.append(f"Estimasi green bean bruto belum dapat dihitung untuk {missing_factor} varian karena rate penyusutan belum tersedia pada tabel sumber.")
        new = sorted(set(tab["varian"]) - set(B["known"]))
        if new:
            warn.append("Varian tidak ada di data latih: model tidak memiliki riwayat khusus varian tersebut; hasil harus divalidasi manual: " + ", ".join(new[:10]))
        last = feat["tanggal"].max()
        hday = feat.groupby("tanggal")["gram_terjual"].sum(min_count=1).sort_index()
        incomplete = feat.groupby("tanggal")["tanggal_tidak_lengkap"].max()
        hday.loc[incomplete[incomplete].index] = np.nan
        hist = hday.tail(28) / 1000
        day = fc.groupby("tanggal")["prediksi"].sum() / 1000
        pre = feat[feat["tanggal"] == last].sort_values("gram_terjual", ascending=False).head(30)
        pre_cols = ["tanggal", "varian", "terjual_unit", "gram_terjual", "dow", "bulan", "akhir_pekan", "gajian", "tgl_kembar", "libur", "lag_7", "rm_28", "tanggal_tidak_lengkap"]
        pre = pre[[c for c in pre_cols if c in pre.columns]].copy()
        pre["tanggal"] = pre["tanggal"].dt.strftime("%Y-%m-%d")
        export_tab = tab[FORECAST_OUTPUT_COLUMNS].copy()
        LAST[session["u"]] = {
            "praproses": feat.assign(tanggal=feat["tanggal"].dt.strftime("%Y-%m-%d"), fase="data baru").drop(columns=["gram_w"], errors="ignore"),
            "prediksi_harian": fc.assign(tanggal=fc["tanggal"].dt.strftime("%Y-%m-%d"), prediksi_gram=fc["prediksi"].round(1))[["tanggal", "varian", "prediksi_gram"]],
            "kebutuhan_stok": export_tab.round(3).rename(columns={"akurasi": "akurasi_persen", "galat_kg": "kesalahan_roasted_kg", "galat_green_kg": "kesalahan_green_kg", "buffer_pct": "stok_pengaman_persen", "permintaan_terlindungi_kg": "kebutuhan_roasted_plus_buffer_kg", "green_terlindungi_kg": "kebutuhan_green_plus_buffer_kg"})}
        def sum_if_complete(col):
            vals = pd.to_numeric(tab[col], errors="coerce")
            return round(float(vals.sum()), 2) if len(vals) and vals.notna().all() else None
        total_variants = int(len(tab))
        green_mapping_complete = bool(tab["faktor"].notna().all())
        T = st["__total__"]
        buffer_values = pd.to_numeric(tab["buffer_pct"], errors="coerce").dropna()
        ringkas = {
            "roasted": sum_if_complete("roasted_kg"),
            "green": sum_if_complete("green_kg"),
            "roasted_buffer": sum_if_complete("permintaan_terlindungi_kg"),
            "green_buffer": sum_if_complete("green_terlindungi_kg"),
            "prediksi": f"{fc['tanggal'].min().date()} s.d. {fc['tanggal'].max().date()}",
            "mode": "manual" if buf is not None else "otomatis",
            "buffer_rata": round(float(buffer_values.mean()), 1) if len(buffer_values) else None,
            "tingkat": round(tl * 100), "akurasi": round(float(T["akurasi"]), 1) if T.get("akurasi") is not None else None,
            "dasar_hz": st.get("__hz__", 30), "buffer_basis_hz": st.get("__hz__", 30),
            "buffer_source": st.get("__buffer_source__", "validasi"),
            "buffer_origins": st.get("__origins__", 0), "buffer_horizon_proxy": bool(hz > 30),
            "galat_roasted": round(float(T["galat_kg"]), 3) if T.get("galat_kg") is not None else None,
            "galat_green": round(float(T["galat_kg"] * (sum_if_complete("green_kg") / max(sum_if_complete("roasted_kg") or 0, 1e-9))), 3) if T.get("galat_kg") is not None and green_mapping_complete else None,
            "total_variants": total_variants,
            "green_mapping_complete": green_mapping_complete,
            "shrinkage_coverage": int(tab["faktor"].notna().sum()),
            "green_provisional_count": int((~tab["status_pemetaan"].astype(str).str.lower().str.startswith("cocok")).sum())}
        rows = json.loads(export_tab.round(3).to_json(orient="records"))
        trend = float((day.mean() / hist.mean() - 1) * 100) if hist.mean() and hist.mean() > 0 else 0.0
        LAST[session["u"]]["ctx"] = {"hz": hz, "ringkas": ringkas, "warn": warn, "tren": trend,
                                      "tabel": [{**r, "bias": st.get(r["varian"], {}).get("bias")} for r in rows[:10]]}
        fk = fc.merge(tab[["varian", "faktor"]], on="varian", how="left")
        fk["roasted"] = fk["prediksi"] / 1000
        fk["green"] = fk["roasted"] * fk["faktor"]
        BLN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
        def per(kunci, nama):
            res = []
            for i, (key, g) in enumerate(fk.groupby(kunci), 1):
                green_sum = g["green"].sum(min_count=len(g))
                roast_sum = g["roasted"].sum()
                dmin, dmax = g["tanggal"].min(), g["tanggal"].max()
                res.append({"periode": nama(i, type("Range", (), {"mulai": dmin, "akhir": dmax})()),
                            "rentang": f"{dmin.day} {BLN[dmin.month - 1][:3]} s.d. {dmax.day} {BLN[dmax.month - 1][:3]} {dmax.year}",
                            "hari": int(g["tanggal"].nunique()), "roasted": round(float(roast_sum), 1),
                            "green": round(float(green_sum), 1) if pd.notna(green_sum) else None,
                            "rata_harian": round(float(roast_sum) / max(int(g["tanggal"].nunique()), 1), 1)})
            return res
        def akurasi_test_horizon(k):
            # Angka tabel ringkasan pada hasil merupakan evaluasi periode uji penelitian;
            # buffer otomatis tetap dikalibrasi terpisah dari periode validasi.
            rt = B["rek"][B["rek"]["hz"] == k]
            if rt.empty:
                return None
            agg = rt.groupby("o", as_index=False)[["gram_terjual", "prediksi"]].sum()
            mt = metrik_rek(agg, 0.90)
            return round(float(mt["akurasi"]), 1) if mt.get("akurasi") is not None else None
        rg = {"mingguan": per((fk["tanggal"] - fk["tanggal"].min()).dt.days // 7, lambda i, r: f"Minggu ke-{i}"),
              "bulanan": per(fk["tanggal"].dt.to_period("M"), lambda i, r: f"{BLN[r.mulai.month - 1]} {r.mulai.year}"),
              "akurasi": {str(k): akurasi_test_horizon(k) for k in (7, 30)}}
        pre = pre.astype(object).where(pd.notna(pre), None)
        return jsonify(log=log + B.get("audit", []) + B.get("audit_penyusutan", []), warn=warn, tabel=rows, pre=pre.to_dict("records"), ringkas=ringkas, ringkasan=rg,
                       horizon=hz, model_source=forecast_B.get("model_source", "model penelitian"),
                       harian={"h": [str(i.date()) for i in hist.index], "hv": [None if pd.isna(v) else round(float(v), 2) for v in hist.values],
                               "f": [str(i.date()) for i in day.index], "fv": day.round(2).tolist()})
    except Exception as e:
        app.logger.exception("Prediksi gagal")
        return jsonify(error=str(e)), 400


@app.route("/api/ai", methods=["POST"])
@login_required
def api_ai():
    c = LAST.get(session["u"], {}).get("ctx")
    if not c:
        return jsonify(error="Jalankan prediksi terlebih dahulu."), 400
    return jsonify(analisis_ai(c))


@app.route("/perbandingan")
@login_required
def perbandingan():
    return render_template("banding.html", page="banding")


@app.route("/api/banding")
@login_required
def api_banding():
    bd = bundle()["banding"]
    return jsonify(banding=bd, kesimpulan=kesimpulan_banding(bd))


@app.route("/api/banding/ai", methods=["POST"])
@login_required
def api_banding_ai():
    bd = bundle()["banding"]
    return jsonify(jawab_ai(SISTEM_BANDING, bd, lambda: kesimpulan_banding(bd)))


@app.route("/api/ai/tes")
@login_required
def api_ai_tes():
    """Uji koneksi AI tanpa menjalankan prediksi: buka /api/ai/tes di browser setelah login."""
    try:
        r = panggil_llm("Balas dengan satu kata saja.", "Tulis: siap")
        return jsonify(ok=bool(r), pesan=(f"Terhubung ke {r[1]} ({r[2]})." if r else "Kunci AI belum diisi (GEMINI_API_KEY)."), jawaban=(r[0][:1] if r else None))
    except Exception as e:
        return jsonify(ok=False, pesan=str(e)[:500])


@app.route("/unduh/<jenis>")
@login_required
def unduh(jenis):
    d = LAST.get(session["u"], {}).get(jenis)
    if d is None:
        return redirect(url_for("prediksi"))
    return Response(d.to_csv(index=False), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={jenis}.csv"})


if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "--fit-operational":
    if len(sys.argv) != 4:
        raise SystemExit("Penggunaan internal: app.py --fit-operational <config_key> <output_path>")
    _fit_operational_cli(sys.argv[2], sys.argv[3])
elif __name__ == "__main__":
    bundle()
    print("Model siap. Buka http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
