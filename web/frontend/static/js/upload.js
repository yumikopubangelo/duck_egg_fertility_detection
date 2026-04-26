HIconst dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const previewBox  = document.getElementById('preview-box');
const previewImg  = document.getElementById('preview-img');
const previewName = document.getElementById('preview-name');
const btnPredict  = document.getElementById('btn-predict');
const resultArea  = document.getElementById('result-area');
const resultBadge = document.getElementById('result-badge');
let selectedFile = null;

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewBox.classList.remove('d-none');
    dropZone.classList.add('d-none');
    btnPredict.disabled = false;
  };
  reader.readAsDataURL(file);
}

function confTier(c) {
  if (c >= 0.90) return { label: 'Sangat Yakin', cls: 'success', desc: 'Model sangat yakin dengan prediksi ini' };
  if (c >= 0.70) return { label: 'Cukup Yakin',  cls: 'warning', desc: 'Model cukup yakin; pertimbangkan monitoring tambahan' };
  return           { label: 'Kurang Yakin', cls: 'danger',  desc: 'Keyakinan rendah — disarankan verifikasi manual' };
}

function recommendation(label, conf) {
  if (label === 'fertile') {
    if (conf >= 0.85) return 'Telur ini sangat direkomendasikan untuk ditetaskan. Indikator vaskularisasi dan embrio terdeteksi dengan kuat.';
    if (conf >= 0.70) return 'Telur ini direkomendasikan untuk ditetaskan. Lakukan monitoring berkala selama masa inkubasi.';
    return 'Model memprediksi fertile namun dengan keyakinan rendah. Disarankan konfirmasi visual atau pengulangan pemeriksaan.';
  } else {
    if (conf >= 0.85) return 'Telur ini tidak direkomendasikan untuk ditetaskan. Tidak terdeteksi tanda-tanda kehidupan embrio.';
    if (conf >= 0.70) return 'Telur ini diprediksi tidak subur. Pertimbangkan untuk tidak diikutsertakan dalam proses inkubasi.';
    return 'Model memprediksi infertile namun dengan keyakinan rendah. Diperlukan pemeriksaan lebih lanjut sebelum mengambil keputusan.';
  }
}

function distanceBar(distances, clusterLabels) {
  if (!distances || distances.length === 0) return '';
  const maxD = Math.max(...distances);
  const rows = distances.map((d, i) => {
    const lbl = clusterLabels[i] || ('Cluster ' + i);
    const isSelected = clusterLabels.selected === i;
    const barW = maxD > 0 ? ((d / maxD) * 100).toFixed(1) : 0;
    const color = lbl === 'fertile' ? '#16a34a' : '#64748b';
    const badge = isSelected ? ('<span class="badge ms-1" style="font-size:.65rem;background:' + color + ';color:#fff">dipilih</span>') : '';
    return '<div class="mb-2">'
      + '<div class="d-flex justify-content-between align-items-center mb-1" style="font-size:.8rem">'
      + '<span class="text-capitalize">' + lbl + badge + '</span>'
      + '<span class="text-muted font-monospace">' + d.toFixed(4) + '</span>'
      + '</div>'
      + '<div class="progress" style="height:7px;border-radius:99px;background:#e2e8f0">'
      + '<div class="progress-bar" style="width:' + barW + '%;border-radius:99px;background:' + color + ';opacity:.7"></div>'
      + '</div></div>';
  }).join('');
  return '<div>' + rows + '<p class="text-muted mb-0" style="font-size:.73rem">Batang lebih pendek = lebih dekat ke pusat cluster</p></div>';
}

btnPredict.addEventListener('click', () => {
  if (!selectedFile) return;
  btnPredict.disabled = true;
  btnPredict.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Menganalisis...';
  resultBadge.className = 'd-none';
  resultArea.innerHTML  = '<div class="text-center py-5"><div class="spinner-border text-primary"></div><p class="mt-3 text-muted small">Memproses gambar dan mengekstrak fitur…</p></div>';

  const fd = new FormData();
  fd.append('file', selectedFile);

  fetch('/api/predict', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(d => {
      if (d.error) throw new Error(d.error);

      const isFertile  = d.prediction === 'fertile';
      const conf       = d.confidence || 0;
      const confPct    = (conf * 100).toFixed(1);
      const purityPct  = ((d.cluster_purity || 0) * 100).toFixed(1);
      const tier       = confTier(conf);
      const colorClass = isFertile ? 'success' : 'danger';
      const icon       = isFertile ? 'check-circle-fill' : 'x-circle-fill';
      const labelID    = isFertile ? 'SUBUR' : 'TIDAK SUBUR';
      const labelEN    = isFertile ? 'Fertile' : 'Infertile';
      const recText    = recommendation(d.prediction, conf);

      const scores = d.label_scores || {};
      const fertileScore   = scores.fertile   != null ? (scores.fertile   * 100).toFixed(1) : null;
      const infertileScore = scores.infertile != null ? (scores.infertile * 100).toFixed(1) : null;

      const clusterLabels = {};
      if (d.distances && d.distances.length === 2) {
        clusterLabels[d.cluster_id] = d.prediction;
        clusterLabels[1 - d.cluster_id] = isFertile ? 'infertile' : 'fertile';
        clusterLabels.selected = d.cluster_id;
      }

      resultBadge.className = 'badge bg-' + colorClass;
      resultBadge.textContent = labelEN;

      let scoreRows = '';
      if (fertileScore != null) {
        scoreRows += '<div class="d-flex justify-content-between align-items-center mb-1" style="font-size:.8rem"><span class="text-success">Fertile</span><strong>' + fertileScore + '%</strong></div>';
        scoreRows += '<div class="progress mb-2" style="height:6px;border-radius:99px;background:#e2e8f0"><div class="progress-bar bg-success" style="width:' + Math.min(100, parseFloat(fertileScore)) + '%;border-radius:99px"></div></div>';
      }
      if (infertileScore != null) {
        scoreRows += '<div class="d-flex justify-content-between align-items-center mb-1" style="font-size:.8rem"><span class="text-danger">Infertile</span><strong>' + infertileScore + '%</strong></div>';
        scoreRows += '<div class="progress" style="height:6px;border-radius:99px;background:#e2e8f0"><div class="progress-bar bg-danger" style="width:' + Math.min(100, parseFloat(infertileScore)) + '%;border-radius:99px"></div></div>';
      }

      const distHTML = distanceBar(d.distances, clusterLabels);
      const imgShape = d.preprocessed_shape ? (d.preprocessed_shape[0] + '×' + d.preprocessed_shape[1]) : '256×256';

      resultArea.innerHTML =
        '<div class="text-center mb-3">'
        + '<i class="bi bi-' + icon + ' text-' + colorClass + '" style="font-size:3rem"></i>'
        + '<h4 class="fw-bold mt-2 mb-0 text-' + colorClass + '">' + labelID + '</h4>'
        + '<p class="text-muted small mb-2">' + labelEN + ' — AWC Clustering</p>'
        + '<span class="badge bg-' + tier.cls + '" style="font-size:.78rem">' + tier.label + '</span>'
        + '</div>'
        + '<hr class="my-3">'
        + '<div class="mb-3">'
        + '<div class="d-flex justify-content-between align-items-center mb-1">'
        + '<span class="fw-semibold" style="font-size:.85rem">Keyakinan Prediksi</span>'
        + '<strong class="text-' + colorClass + '">' + confPct + '%</strong>'
        + '</div>'
        + '<div class="progress mb-1" style="height:10px;border-radius:99px;background:#e2e8f0">'
        + '<div class="progress-bar bg-' + colorClass + '" style="width:' + confPct + '%;border-radius:99px"></div>'
        + '</div>'
        + '<p class="text-muted mb-0" style="font-size:.77rem">' + tier.desc + '</p>'
        + '</div>'
        + '<div class="mb-3"><div class="fw-semibold mb-2" style="font-size:.85rem">Distribusi Probabilitas Kelas</div>' + scoreRows + '</div>'
        + (distHTML ? '<div class="mb-3"><div class="fw-semibold mb-2" style="font-size:.85rem">Jarak ke Pusat Cluster (Ruang Fitur)</div>' + distHTML + '</div>' : '')
        + '<div class="alert alert-' + colorClass + ' py-2 px-3 mb-3" role="alert" style="font-size:.82rem"><strong>Rekomendasi:</strong> ' + recText + '</div>'
        + '<div>'
        + '<button class="btn btn-outline-secondary btn-sm w-100" type="button" data-bs-toggle="collapse" data-bs-target="#techDetail">Detail Teknis</button>'
        + '<div class="collapse mt-2" id="techDetail">'
        + '<div class="rounded-3 p-3" style="background:#f8fafc;font-size:.8rem"><div class="row g-1">'
        + '<div class="col-5 text-muted">File</div><div class="col-7 fw-semibold text-truncate">' + (d.original_filename || selectedFile.name) + '</div>'
        + '<div class="col-5 text-muted">Metode</div><div class="col-7">Adaptive Weighted Clustering</div>'
        + '<div class="col-5 text-muted">Cluster ID</div><div class="col-7">#' + d.cluster_id + ' &middot; purity ' + purityPct + '%</div>'
        + '<div class="col-5 text-muted">Fitur diekstrak</div><div class="col-7">' + (d.feature_count || 70) + ' fitur</div>'
        + '<div class="col-5 text-muted">Ukuran input</div><div class="col-7">' + imgShape + ' px</div>'
        + '<div class="col-5 text-muted">ID Prediksi</div><div class="col-7 font-monospace">' + (d.id || '-').substring(0,8) + '…</div>'
        + '<div class="col-5 text-muted">Waktu</div><div class="col-7">' + (d.timestamp ? new Date(d.timestamp).toLocaleString('id-ID') : '-') + '</div>'
        + '</div></div></div></div>'
        + '<div class="mt-3 d-flex flex-column gap-2">'
        + '<button class="btn btn-outline-info btn-sm w-100" id="btn-segment" onclick="runSegmentation()">'
        + '<i class="bi bi-grid-3x3 me-1"></i>Lihat Segmentasi U-Net</button>'
        + '<div id="segment-area"></div>'
        + '<button class="btn btn-outline-secondary btn-sm w-100" onclick="resetUpload()">'
        + '<i class="bi bi-arrow-left me-1"></i>Prediksi Gambar Lain</button>'
        + '</div>';
    })
    .catch(err => {
      resultBadge.className = 'd-none';
      resultArea.innerHTML = '<div class="text-center text-danger py-4">'
        + '<i class="bi bi-exclamation-circle fs-2 d-block mb-2"></i>'
        + '<p class="mb-0">Gagal melakukan prediksi.<br><small class="text-muted">' + err.message + '</small></p>'
        + '</div>';
    })
    .finally(() => {
      btnPredict.disabled = false;
      btnPredict.innerHTML = '<i class="bi bi-search me-2"></i>Analisis Kesuburan';
    });
});

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewBox.classList.add('d-none');
  dropZone.classList.remove('d-none');
  btnPredict.disabled = true;
  resultBadge.className = 'd-none';
  resultArea.innerHTML = '<div class="text-center text-muted py-5">'
    + '<i class="bi bi-hourglass fs-1 d-block mb-2 opacity-50"></i>'
    + '<p class="mb-0">Hasil akan muncul di sini setelah analisis</p>'
    + '</div>';
}

/* ── U-Net Segmentation ──────────────────────────────────── */
function runSegmentation() {
  if (!selectedFile) return;
  const btn = document.getElementById('btn-segment');
  const area = document.getElementById('segment-area');
  if (!btn || !area) return;

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Menjalankan U-Net…';
  area.innerHTML = '';

  const fd = new FormData();
  fd.append('file', selectedFile);

  fetch('/api/segment', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(d => {
      if (d.error) throw new Error(d.error);

      const areas  = d.class_areas || {};
      const colors = d.class_colors || {};
      const bgPct  = (areas.background || 0).toFixed(1);

      // Build class area badges dynamically from API response
      let areaBadges =
        '<span style="background:#1e293b;color:#fff;padding:2px 8px;border-radius:99px">Background ' + bgPct + '%</span>';
      if (d.n_classes === 1) {
        const eggPct = (areas.egg_region || 0).toFixed(1);
        areaBadges +=
          ' <span style="background:#16a34a;color:#fff;padding:2px 8px;border-radius:99px">Egg Region ' + eggPct + '%</span>';
      } else {
        const vasPct = (areas.vascularization || 0).toFixed(1);
        const embPct = (areas.embryo          || 0).toFixed(1);
        areaBadges +=
          ' <span style="background:#dc2626;color:#fff;padding:2px 8px;border-radius:99px">Vaskularisasi ' + vasPct + '%</span>'
          + ' <span style="background:#16a34a;color:#fff;padding:2px 8px;border-radius:99px">Inti Embrio ' + embPct + '%</span>';
      }

      area.innerHTML =
        '<div class="rounded-3 overflow-hidden border mt-2">'
        + '<div class="row g-0">'
        + '<div class="col-6"><img src="' + d.original_b64 + '" class="img-fluid w-100" style="display:block" alt="Original"/>'
        + '<div class="text-center py-1 bg-light" style="font-size:.72rem;color:#64748b">Gambar Asli (256×256)</div></div>'
        + '<div class="col-6"><img src="' + d.overlay_b64 + '" class="img-fluid w-100" style="display:block" alt="Segmentasi"'
        + ' id="seg-overlay-img"/>'
        + '<div class="text-center py-1 bg-light" style="font-size:.72rem;color:#64748b">Overlay Segmentasi</div></div>'
        + '</div></div>'
        + '<div class="d-flex gap-2 mt-2 flex-wrap" style="font-size:.78rem">'
        + areaBadges
        + '</div>'
        + '<a id="dl-seg" href="' + d.overlay_b64 + '" download="segmentasi_unet.png"'
        + ' class="btn btn-outline-secondary btn-sm w-100 mt-2">'
        + '<i class="bi bi-download me-1"></i>Download Overlay</a>';

      btn.innerHTML = '<i class="bi bi-grid-3x3 me-1"></i>Perbarui Segmentasi';
    })
    .catch(err => {
      area.innerHTML =
        '<div class="alert alert-danger py-2 px-3 mt-2" style="font-size:.8rem">'
        + '<i class="bi bi-exclamation-circle me-1"></i>' + err.message + '</div>';
      btn.innerHTML = '<i class="bi bi-grid-3x3 me-1"></i>Lihat Segmentasi U-Net';
    })
    .finally(() => { btn.disabled = false; });
}
