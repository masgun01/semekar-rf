"""Focused regression tests for the Semekar forecasting/inventory pipeline.

Run after installing requirements with: python -m unittest discover -s tests -v
"""
import os
import unittest
import numpy as np
import pandas as pd

import app


class MetricsTests(unittest.TestCase):
    def test_wape_and_zero_target_mape_handling(self):
        got = app.metrik([0, 0, 10], [1, 0, 8])
        self.assertAlmostEqual(got["MAE"], 1.0)
        self.assertAlmostEqual(got["WAPE"], 30.0)
        self.assertAlmostEqual(got["MAPE"], 20.0)  # zero-target rows are excluded from MAPE only
        self.assertEqual(got["n"], 3)

    def test_buffer_uses_validation_calibration_not_test_backtest(self):
        # Backtest test memiliki buffer 90%, tetapi validasi hanya 20%.
        # Statistik yang dipakai untuk buffer operasional harus membaca validasi.
        test = pd.DataFrame({
            "hz": [7], "varian": ["V1"], "prediksi": [100.0],
            "gram_terjual": [190.0], "o": [pd.Timestamp("2025-10-01")]
        })
        calibration = pd.DataFrame({
            "hz": [7], "varian": ["V1"], "prediksi": [100.0],
            "gram_terjual": [120.0], "o": [pd.Timestamp("2025-07-01")]
        })
        got = app.statistik({"rek": test, "buffer_calibration": calibration}, 7, 0.90)
        self.assertAlmostEqual(got["V1"]["buffer"], 20.0)
        self.assertEqual(got["__buffer_source__"], "validasi")
        self.assertEqual(got["__origins__"], 1)


class FeatureTests(unittest.TestCase):
    def test_lags_use_previous_observations(self):
        d = pd.date_range("2025-01-01", periods=35, freq="D")
        f = pd.DataFrame({"tanggal": d, "varian": "V1", "gram_feature": np.arange(35, dtype=float)})
        x = app.add_features(f, target_col="gram_feature")
        self.assertTrue(np.isnan(x.loc[0, "lag_1"]))
        self.assertEqual(x.loc[10, "lag_1"], 9.0)
        self.assertEqual(x.loc[10, "lag_7"], 3.0)
        self.assertEqual(x.loc[10, "rm_7"], 6.0)  # mean of values 3..9, not including current day 10


class InventoryTests(unittest.TestCase):
    def test_stock_recommendation_subtracts_on_hand_and_confirmed_supply(self):
        B = {"faktor": {"V1": 1.2}}
        inv_row = {"varian": "V1", "varian_key": "V1"}
        inv_row.update({
            "stok_produk_jadi_kg": 2.0,
            "pasokan_produk_jadi_kg": 0.0,
            "target_akhir_produk_jadi_kg": 1.0,
            "stok_roasted_kg": 3.0,
            "pasokan_roasted_kg": 1.0,
            "target_akhir_roasted_kg": 2.0,
            "stok_green_kg": 2.0,
            "pasokan_green_kg": 1.0,
            "target_akhir_green_kg": 1.0,
        })
        inv = pd.DataFrame([inv_row])
        fc = pd.DataFrame({"varian": ["V1"], "prediksi": [10000.0]})
        stats = {"__total__": {"akurasi": 90, "galat_kg": 1, "buffer": 0},
                 "V1": {"akurasi": 90, "galat_kg": 1, "buffer": 0}}
        row = app.kebutuhan(fc, B, 0, stats, inventory=inv).iloc[0]
        self.assertAlmostEqual(row["produksi_produk_jadi_kg"], 9.0)
        self.assertAlmostEqual(row["roasted_perlu_diproduksi_kg"], 7.0)
        self.assertAlmostEqual(row["green_perlu_bruto_kg"], 8.4)
        self.assertAlmostEqual(row["rekomendasi_green_beli_kg"], 6.4)


class ResearchDataTests(unittest.TestCase):
    def test_research_frame_excludes_incomplete_imputed_date(self):
        frame, audit = app._create_research_frame()
        self.assertEqual(frame["varian"].nunique(), 27)
        bad = frame[frame["tanggal"].eq(pd.Timestamp("2025-11-30"))]
        self.assertEqual(len(bad), 27)
        self.assertTrue(bad["tanggal_tidak_lengkap"].all())
        self.assertFalse(bad["label_valid"].any())
        self.assertTrue(any("imputasi" in x.lower() for x in audit))

    def test_operational_update_rejects_short_history(self):
        d = pd.date_range("2025-01-01", periods=45, freq="D")
        f = pd.DataFrame({"tanggal": d, "varian": "V1", "gram_terjual": np.arange(45, dtype=float),
                          "gram_feature": np.arange(45, dtype=float), "label_valid": True,
                          "tanggal_tidak_lengkap": False})
        f = app.add_features(f, target_col="gram_feature")
        config = app.FEATURE_CONFIGS[1]
        B = {"selected_config": config}
        with self.assertRaisesRegex(ValueError, "90 tanggal"):
            app._validate_and_fit_operational(f, B)


class ShrinkageSourceTests(unittest.TestCase):
    def test_all_sampled_variants_get_shrinkage_from_stock_workbook(self):
        shrink = app.load_variant_shrinkage()
        self.assertEqual(shrink["jumlah_rate"], 27)
        self.assertAlmostEqual(shrink["penyusutan"]["ROBUSTA TORAJA"], 0.23)
        self.assertAlmostEqual(shrink["faktor"]["ROBUSTA TORAJA"], 1 / 0.77)
        self.assertAlmostEqual(shrink["penyusutan"]["ARABICA GAYO WINE"], 0.17)
        self.assertIn("Stock.xlsx", " ".join(shrink["audit_penyusutan"]))

    def test_green_bean_estimate_is_returned_even_if_product_mapping_is_provisional(self):
        shrink = app.load_variant_shrinkage()
        B = {**shrink}
        fc = pd.DataFrame({"varian": ["ARABICA GAYO WINE"], "prediksi": [10000.0]})
        stats = {"__total__": {"akurasi": 90, "galat_kg": 1, "buffer": 0},
                 "ARABICA GAYO WINE": {"akurasi": 90, "galat_kg": 1, "buffer": 0}}
        row = app.kebutuhan(fc, B, 0, stats).iloc[0]
        self.assertAlmostEqual(row["roasted_g"], 10000)
        self.assertAlmostEqual(row["penyusutan_pct"], 17)
        self.assertAlmostEqual(row["green_g"], 10000 / (1 - 0.17))
        self.assertIn("konfirmasi", row["status_pemetaan"].lower())

    def test_forecast_bruto_and_buffer_do_not_require_inventory_file(self):
        shrink = app.load_variant_shrinkage()
        fc = pd.DataFrame({"varian": ["ARABICA GAYO WINE"], "prediksi": [10000.0]})
        stats = {"__total__": {"akurasi": 90, "galat_kg": 1, "buffer": 0},
                 "ARABICA GAYO WINE": {"akurasi": 90, "galat_kg": 1, "buffer": 0}}
        # inventory omitted deliberately: gross roast/green estimates must still be present.
        row = app.kebutuhan(fc, shrink, 10, stats).iloc[0]
        self.assertAlmostEqual(row["roasted_kg"], 10.0)
        self.assertAlmostEqual(row["green_kg"], 10.0 / (1 - 0.17))
        self.assertAlmostEqual(row["permintaan_terlindungi_kg"], 11.0)
        self.assertAlmostEqual(row["green_terlindungi_kg"], 11.0 / (1 - 0.17))
        self.assertFalse(bool(row["stok_tersedia"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
