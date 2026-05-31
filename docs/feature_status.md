# Status Fitur dan Kesesuaian dengan Penelitian Pak Andi

Dokumen ini merangkum status implementasi project berdasarkan kondisi codebase saat ini, lalu membandingkannya dengan arah penelitian Pak Andi: `preprocessing -> U-Net multiclass -> fitur klasik + deep embedding -> AWC -> pembandingan dengan baseline`.

Fokus dokumen ini bukan hanya "sudah ada atau belum", tetapi juga:

- apakah implementasi sudah sesuai dengan rancangan penelitian,
- apa yang masih perlu ditingkatkan,
- apa yang belum ada dan perlu ditambahkan agar lebih konsisten dengan penelitian Pak Andi.

## Ringkasan Eksekutif

Secara umum, project ini **sudah jauh lebih dekat** ke penelitian Pak Andi dibanding versi awal. Bagian yang sudah kuat:

- preprocessing sudah ada,
- U-Net multiclass sudah ada,
- fitur klasik penting seperti `GLCM` dan `morfologi vaskular` sudah ada,
- AWC sudah aktif dan saat ini menjadi model terbaik,
- baseline `K-Means`, `FCM`, dan `MobileNetV3` sudah masuk ke evaluasi.

Namun masih ada beberapa celah penting agar benar-benar rapi secara akademik:

1. **Evaluasi U-Net belum sepenuhnya konsisten dengan setup multiclass.**
2. **Definisi kelas segmentasi masih perlu dikunci final dan diselaraskan dengan naskah.**
3. **Deep embedding bottleneck U-Net sudah ada, tetapi belum menjadi pipeline aktif terbaik untuk AWC.**
4. **Laporan hasil U-Net masih perlu diperkuat dengan metrik per kelas dan paket evaluasi final yang lebih siap untuk BAB IV.**

## 1. Status U-Net Saat Ini

### 1.1 Yang Sudah Ada

- **Arsitektur U-Net tersedia dan aktif**
  - Lokasi: `src/segmentation/unet.py`, `src/segmentation/unet_lightweight.py`
- **Training U-Net tersedia**
  - Lokasi: `scripts/03_train_unet.py`
- **Konfigurasi final saat ini sudah multiclass**
  - `lightweight: true`
  - `n_classes: 3`
  - `loss_type: ce`
  - Lokasi: `configs/unet_config.yaml`
- **Dataset segmentasi multiclass memang sudah ada**
  - mask bernilai `0, 1, 2`
  - Lokasi: `data/segmentation/*/masks`
- **Checkpoint U-Net nyata tersedia**
  - banyak checkpoint di `results/unet_training/checkpoints/`
  - `models/unet/model.pth` terisi
- **Inference segmentasi di web tersedia**
  - Lokasi: `web/api/routes/segmentation.py`

### 1.2 Penilaian Terhadap Penelitian Pak Andi

Untuk penelitian Pak Andi, U-Net berfungsi sebagai:

- pemisah area biologis yang relevan,
- penyedia mask untuk analisis pasca-segmentasi,
- sumber **deep embedding bottleneck** untuk pipeline hybrid.

Dalam konteks itu, status U-Net sekarang bisa dibilang:

- **sudah layak sebagai fondasi eksperimen**,  
- tetapi **belum optimal sebagai paket hasil segmentasi final untuk naskah disertasi/skripsi**.

### 1.3 Yang Perlu Ditingkatkan pada U-Net

1. **Evaluator multiclass harus dirapikan**
   - `scripts/06_evaluate_models.py` masih memiliki logika biner `sigmoid > 0.5` pada bagian evaluasi segmentasi.
   - Ini tidak ideal untuk setup U-Net `n_classes=3`.
   - Yang perlu dilakukan:
     - ubah evaluasi agar konsisten menggunakan `argmax` untuk multiclass,
     - laporkan `mean IoU`, `mean Dice`, dan **IoU/Dice per kelas**.

2. **Perlu paket evaluasi U-Net yang siap BAB IV**
   Saat ini training ada, model ada, endpoint segmentasi ada, tetapi paket laporan segmentasi belum setegas paket evaluasi klasifikasi.  
   Yang perlu ditambahkan:
   - confusion matrix pixel-level,
   - Dice per kelas,
   - IoU per kelas,
   - visualisasi prediksi terbaik, sedang, terburuk,
   - ringkasan hasil final khusus segmentasi.

3. **Kelas segmentasi final harus dikunci**
   Saat ini codebase aktif cenderung memakai:
   - `0 = background`
   - `1 = vascularization`
   - `2 = embryo`

   Ini sudah konsisten di:
   - `web/api/routes/segmentation.py`
   - `scripts/03b_generate_masks.py`

   Tapi di naskah penelitian sempat muncul variasi lain seperti:
   - `yolk, albumen, vaskularisasi`, atau
   - `region biologis multikelas`.

   Jadi yang perlu dilakukan bukan hanya coding, tapi **menentukan satu definisi operasional final** lalu menyelaraskan BAB III dan BAB IV ke definisi itu.

4. **Artefak model U-Net masih perlu dibersihkan**
   - `models/unet/model.pth` valid
   - `models/unet/unet_best.pth` masih `0 byte`

   Ini kecil, tapi penting untuk kebersihan eksperimen dan reproducibility.

5. **Post-processing segmentasi masih bisa diperkuat**
   - `src/segmentation/postprocessing.py` masih minim.
   - Untuk penelitian Pak Andi, post-processing bisa ditingkatkan dengan:
     - connected component filtering,
     - class-specific cleanup,
     - smoothing tepi mask,
     - validasi area biologis yang tidak masuk akal.

## 2. Kesesuaian Fitur dengan Penelitian Pak Andi

### 2.1 Fitur yang Sudah Sesuai atau Sudah Mendekati Sesuai

#### Preprocessing
- Sudah ada pipeline preprocessing.
- Status: **sesuai secara umum**, tinggal dikunci pada jalur final yang dipakai untuk eksperimen utama.

#### GLCM
- Sudah ditambahkan ke fitur klasik.
- Lokasi: `src/features/classical_features.py`
- Status: **sesuai dan penting**, karena ini salah satu fitur vital menurut proposal.

#### Morfologi vaskular
- Sudah ditambahkan:
  - `vascular_skeleton_total_length`
  - `vascular_node_count`
  - `vascular_branch_density`
- Status: **baru sesuai** dengan proposal, sebelumnya belum.

#### Baseline MobileNetV3
- Sudah ditambahkan sebagai baseline end-to-end CNN.
- Lokasi:
  - `src/classification/mobilenet_v3_baseline.py`
  - `scripts/run_full_evaluation.py`
- Status: **sudah sesuai**, walau masih bisa diperkuat dengan training yang lebih matang.

### 2.2 Fitur yang Sudah Ada di Kode tetapi Belum Menjadi Jalur Utama

#### Deep embedding dari bottleneck U-Net via GAP
- Implementasi sudah ada:
  - `src/features/deep_features.py`
  - `src/features/hybrid_features.py`
- Eksperimen hybrid juga sudah pernah dijalankan.

Tetapi status aktualnya:

- **sudah ada secara implementasi**
- **belum menjadi pipeline aktif terbaik**
- **belum menjadi model utama yang dipakai aplikasi**

Saat ini model AWC utama yang aktif justru memakai:

- fitur klasik + GLCM + morfologi vaskular,
- bukan hybrid + deep embedding.

Jadi untuk penelitian Pak Andi, bagian ini statusnya:

- **belum final secara ilmiah**,  
- bukan karena tidak ada, tetapi karena **hasilnya belum mengungguli pipeline utama**.

## 3. Hasil yang Saat Ini Paling Kuat

Pipeline yang saat ini paling kuat di codebase adalah:

- preprocessing
- U-Net multiclass sebagai pendukung segmentasi
- fitur klasik + GLCM + morfologi vaskular
- AWC dengan seleksi fitur ANOVA top-20

Hasil evaluasi utamanya saat ini:

- **AWC Accuracy = 0.9167**
- **F1 = 0.9231**
- **ROC-AUC = 0.9514**

Lokasi:
- `results/evaluation/full_vascular_glcm_mobilenet/comparison_table.csv`

Visual ringkas hasil evaluasi:

![Ringkasan evaluasi model](./ringkasan_evaluasi_model.png)

Ini berarti:

- secara performa, pipeline utama sudah kuat,
- tetapi secara kesesuaian penuh dengan narasi penelitian Pak Andi, **jalur deep embedding dan laporan segmentasi U-Net masih perlu dibereskan**.

## 4. Yang Belum dan Harus Ditambahkan agar Lebih Sesuai dengan Pak Andi

### Prioritas Tinggi

1. **Perbaiki evaluasi U-Net multiclass**
   - update `scripts/06_evaluate_models.py`
   - tambahkan metrik multiclass yang benar

2. **Buat laporan segmentasi final**
   - mean Dice
   - mean IoU
   - Dice per kelas
   - IoU per kelas
   - contoh visual hasil segmentasi

3. **Kunci definisi kelas segmentasi final**
   - tentukan apakah finalnya:
     - `background, vascularization, embryo`
     - atau definisi biologis lain yang disetujui pembimbing

4. **Tentukan posisi deep embedding dalam naskah**
   Ada dua opsi yang sama-sama jujur:
   - **Opsi A:** tetap sebagai novelty/eksperimen tambahan, tetapi bukan model utama karena hasilnya belum terbaik
   - **Opsi B:** lanjut tuning hybrid sampai benar-benar kompetitif

### Prioritas Menengah

1. **Training MobileNetV3 baseline yang lebih matang**
   - epoch lebih panjang
   - opsi pretrained
   - agar pembanding CNN end-to-end lebih fair

2. **Pembersihan artefak model**
   - hapus/arsipkan file `0 byte`
   - pastikan satu checkpoint final yang resmi

3. **Peningkatan post-processing U-Net**
   - per kelas
   - lebih stabil untuk mask biologis

## 5. Jawaban Singkat untuk Pertanyaan "U-Net Saya Masih Perlu Dikembangkan?"

**Ya, masih perlu, tetapi bukan dari nol.**

Yang perlu dikembangkan bukan arsitektur dasarnya dulu, melainkan:

1. **evaluasi multiclass-nya**,  
2. **pelaporan hasilnya**,  
3. **konsistensi kelas segmentasi dengan naskah**,  
4. **pemanfaatan bottleneck embedding jika memang ingin dipertahankan sebagai novelty utama**.

Jadi posisi U-Net sekarang adalah:

- **sudah cukup matang sebagai backbone segmentasi**,  
- tetapi **belum selesai sebagai komponen penelitian yang siap ditulis final tanpa revisi tambahan**.

## 6. Prioritas Pengembangan yang Disarankan

Urutan kerja yang paling aman agar tetap selaras dengan penelitian Pak Andi:

1. rapikan evaluator U-Net multiclass  
2. hasilkan paket evaluasi segmentasi final  
3. kunci definisi kelas segmentasi di kode dan dokumen  
4. putuskan status akhir deep embedding:
   - ditingkatkan,
   - atau diposisikan sebagai eksperimen pembanding/ablation
5. jika perlu, lakukan run MobileNetV3 baseline yang lebih kuat untuk pembanding final

---

Dokumen ini perlu diperbarui lagi setiap kali ada perubahan besar pada:

- pipeline segmentasi,
- feature extractor,
- baseline CNN,
- atau hasil evaluasi final yang akan dibawa ke BAB IV.
