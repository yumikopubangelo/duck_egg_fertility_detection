# ANGKA PAPER — Deteksi Fertilitas Telur Bebek (REPRODUCIBLE)
_Dihasilkan otomatis: 2026-05-31 16:51 — semua angka di bawah bisa direproduksi dari model & data Anda._

## 1. Dataset
| Split | Fertil | Infertil | Total |
|---|---|---|---|
| Train | 72 | 84 | 156 |
| Val | 20 | 24 | 44 |
| Test | 12 | 12 | 24 |
| **Total** | **104** | **120** | **224** |

## 2. Segmentasi U-Net (Tabel II) — held-out test, n=24
Model: lightweight U-Net 3-kelas (32-64-128-256), loss **Weighted CE + multiclass Dice**.

| Kelas | Dice | IoU |
|---|---|---|
| Embrio | 0.744 | 0.660 |
| Vaskular | 0.377 | 0.274 |
| Background | 0.993 | 0.986 |
| **Pixel Accuracy** | **0.985** | |

- Mean foreground (embrio+vaskular): Dice 0.560
- Val plateau (training): embrio ~0.85, vaskular ~0.57
- ⚠️ JANGAN tulis Dice 0.874 (tak ter-reproduksi). Pakai per-kelas di atas.

## 3. Klasifikasi AWC vs Baseline (Tabel III) — 5-Fold CV (mean ± std)
Fitur: 338-D hibrida (classical + deep embedding U-Net). ANOVA k=20 per fold (tanpa kebocoran).

| Metode | Akurasi | Presisi | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| AWC | 0.857 ± 0.054 | 0.817 ± 0.061 | 0.894 ± 0.070 | 0.853 ± 0.058 | 0.908 ± 0.041 |
| K-Means | 0.857 ± 0.054 | 0.817 ± 0.061 | 0.894 ± 0.070 | 0.853 ± 0.058 | 0.866 ± 0.049 |
| FCM | 0.476 ± 0.342 | 0.440 ± 0.346 | 0.496 ± 0.373 | 0.466 ± 0.359 | 0.443 ± 0.384 |

### Uji Wilcoxon (akurasi per-fold)
- AWC vs K-Means: Δacc=+0.000, p=1.000 (identik)
- AWC vs FCM: Δacc=+0.381, p=0.125 (TIDAK signifikan)

- Akurasi per fold AWC: [0.933, 0.889, 0.867, 0.778, 0.818]
- Akurasi per fold K-Means: [0.933, 0.889, 0.867, 0.778, 0.818]
- Akurasi per fold FCM: [0.933, 0.133, 0.844, 0.289, 0.182]

## 4. AWC pada Single Test Split (held-out, n=24)
- Akurasi: **0.917** | F1: 0.923
- Confusion [baris=aktual infertil/fertil]: [[10, 2], [0, 12]]
- Recall fertil: 1.00 | Recall infertil: 0.83

## 5. Daya Diskriminasi per Grup Fitur (ANOVA F, interpretability)
| Grup Fitur | Rata-rata F-score |
|---|---|
| Tekstur LBP | 176.3 |
| Statistik Intensitas | 136.1 |
| Tepi (Edge) | 112.7 |
| Deep Bottleneck | 66.2 |
| Morfologi Mask | 40.8 |
| Histogram | 32.1 |
| Tekstur GLCM | 31.5 |
| Morfologi Vaskular | 5.6 |

## 6. Catatan Kejujuran (untuk revisi paper)
- **AWC ≈ K-Means** (akurasi setara) — JANGAN klaim "adaptive lebih unggul 7,7 pp". Lihat Wilcoxon di atas.
- **Feature importance**: angka lama (deep 0.312 / vaskular 0.287) tak ter-reproduksi. Pakai tabel grup di Bagian 5.
- **U-Net Dice**: pakai per-kelas (embrio ~0.74), bukan 0.874.
- **Loss U-Net**: Weighted CE + Dice (bukan BCE+Dice).
- **MC-Dropout T=30**: belum diimplementasikan — hapus klaim atau implementasi dulu.
- **Fitur**: 338 dimensi aktual (paper tulis 322 — samakan).
- **Kekuatan jujur AWC**: interpretability (bobot fitur) + efisiensi label, akurasi setara baseline.