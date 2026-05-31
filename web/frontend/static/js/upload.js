let selectedFile = null;

function initUploadPage() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const previewBox = document.getElementById("preview-box");
  const previewImg = document.getElementById("preview-img");
  const previewName = document.getElementById("preview-name");
  const btnPredict = document.getElementById("btn-predict");
  const btnResetPreview = document.getElementById("btn-reset-preview");
  const resultArea = document.getElementById("result-area");
  const resultBadge = document.getElementById("result-badge");

  if (!dropZone || !fileInput || !previewBox || !previewImg || !previewName || !btnPredict || !resultArea || !resultBadge) {
    return;
  }

  const state = {
    dropZone,
    fileInput,
    previewBox,
    previewImg,
    previewName,
    btnPredict,
    btnResetPreview,
    resultArea,
    resultBadge,
  };

  const preventDefaults = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    window.addEventListener(eventName, preventDefaults, false);
  });

  dropZone.addEventListener("dragover", () => dropZone.classList.add("drag-over"));
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (event) => {
    dropZone.classList.remove("drag-over");
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (file) {
      setFile(file, state);
    }
  });

  dropZone.addEventListener("click", (event) => {
    if (event.target.closest("[data-role='file-picker']")) {
      return;
    }
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
    if (file) {
      setFile(file, state);
    }
  });

  if (btnResetPreview) {
    btnResetPreview.addEventListener("click", () => resetUpload(state));
  }

  btnPredict.addEventListener("click", () => predictSelected(state));
  resultArea.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-action='reset-upload']");
    if (trigger) {
      resetUpload(state);
    }
  });

  window.resetUpload = () => resetUpload(state);
}

function setFile(file, state) {
  if (!file || !file.type.startsWith("image/")) {
    renderError(state.resultArea, "File harus berupa gambar.");
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (event) => {
    state.previewImg.src = event.target.result;
    state.previewName.textContent = file.name;
    state.previewBox.classList.remove("d-none");
    state.dropZone.classList.add("d-none");
    state.btnPredict.disabled = false;
  };
  reader.readAsDataURL(file);
}

function resetUpload(state) {
  selectedFile = null;
  state.fileInput.value = "";
  state.previewBox.classList.add("d-none");
  state.dropZone.classList.remove("d-none");
  state.btnPredict.disabled = true;
  state.resultBadge.className = "d-none";
  state.resultArea.innerHTML = [
    '<div class="text-center text-muted py-5">',
    '<i class="bi bi-hourglass fs-1 d-block mb-2 opacity-50"></i>',
    "<p class=\"mb-0\">Hasil akan muncul di sini setelah analisis</p>",
    "</div>",
  ].join("");
}

function predictSelected(state) {
  if (!selectedFile) {
    return;
  }

  state.btnPredict.disabled = true;
  state.btnPredict.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Menganalisis...';
  state.resultBadge.className = "d-none";
  state.resultArea.innerHTML = [
    '<div class="text-center py-5">',
    '<div class="spinner-border text-primary"></div>',
    '<p class="mt-3 text-muted small">Memproses gambar, segmentasi vaskular, dan pembanding model...</p>',
    "</div>",
  ].join("");

  const formData = new FormData();
  formData.append("file", selectedFile);

  fetch("/api/predict", { method: "POST", body: formData })
    .then((response) => response.json())
    .then((payload) => {
      if (payload.error) {
        throw new Error(payload.error);
      }
      renderPrediction(payload, state);
    })
    .catch((error) => {
      state.resultBadge.className = "d-none";
      renderError(state.resultArea, error.message || "Gagal melakukan prediksi.");
    })
    .finally(() => {
      state.btnPredict.disabled = false;
      state.btnPredict.innerHTML = '<i class="bi bi-search me-2"></i>Analisis Kesuburan';
    });
}

function renderError(container, message) {
  container.innerHTML = [
    '<div class="text-center text-danger py-4">',
    '<i class="bi bi-exclamation-circle fs-2 d-block mb-2"></i>',
    `<p class="mb-0">Gagal melakukan prediksi.<br><small class="text-muted">${escapeHtml(message)}</small></p>`,
    "</div>",
  ].join("");
}

function renderPrediction(data, state) {
  const isFertile = data.prediction === "fertile";
  const confidence = Number(data.confidence || 0);
  const confidencePct = (confidence * 100).toFixed(1);
  const purityPct = ((data.cluster_purity || 0) * 100).toFixed(1);
  const tier = confidenceTier(confidence);
  const colorClass = isFertile ? "success" : "danger";
  const icon = isFertile ? "check-circle-fill" : "x-circle-fill";
  const labelId = isFertile ? "SUBUR" : "TIDAK SUBUR";
  const labelEn = isFertile ? "Fertile" : "Infertile";

  state.resultBadge.className = `badge bg-${colorClass}`;
  state.resultBadge.textContent = labelEn;

  const comparisonHtml = renderComparisonTable(data.model_comparison || {});
  const segmentationHtml = renderSegmentationSection(data.segmentation || {});
  const explanationHtml = renderExplanationSection(data.explanation || {});
  const scoreHtml = renderScoreBars(data.label_scores || {});
  const distanceHtml = renderDistanceBars(data.distances || [], data.cluster_id, data.prediction);

  const imgShape = Array.isArray(data.preprocessed_shape)
    ? `${data.preprocessed_shape[0]}x${data.preprocessed_shape[1]}`
    : "256x256";

  state.resultArea.innerHTML = [
    '<div class="text-center mb-3">',
    `<i class="bi bi-${icon} text-${colorClass}" style="font-size:3rem"></i>`,
    `<h4 class="fw-bold mt-2 mb-0 text-${colorClass}">${labelId}</h4>`,
    `<p class="text-muted small mb-2">${labelEn} - AWC Clustering</p>`,
    `<span class="badge bg-${tier.cls}" style="font-size:.78rem">${tier.label}</span>`,
    "</div>",
    '<hr class="my-3">',
    '<div class="mb-3">',
    '<div class="d-flex justify-content-between align-items-center mb-1">',
    '<span class="fw-semibold" style="font-size:.85rem">Keyakinan Prediksi</span>',
    `<strong class="text-${colorClass}">${confidencePct}%</strong>`,
    "</div>",
    '<div class="progress mb-1" style="height:10px;border-radius:99px;background:#e2e8f0">',
    `<div class="progress-bar bg-${colorClass}" style="width:${confidencePct}%;border-radius:99px"></div>`,
    "</div>",
    `<p class="text-muted mb-0" style="font-size:.77rem">${tier.desc}</p>`,
    "</div>",
    scoreHtml,
    distanceHtml,
    explanationHtml,
    segmentationHtml,
    comparisonHtml,
    '<div class="rounded-3 p-3 mt-3" style="background:#f8fafc;font-size:.8rem">',
    '<div class="row g-1">',
    `<div class="col-5 text-muted">File</div><div class="col-7 fw-semibold text-truncate">${escapeHtml(data.original_filename || selectedFile.name)}</div>`,
    '<div class="col-5 text-muted">Metode utama</div><div class="col-7">Adaptive Weighted Clustering</div>',
    `<div class="col-5 text-muted">Cluster ID</div><div class="col-7">#${data.cluster_id} · purity ${purityPct}%</div>`,
    `<div class="col-5 text-muted">Fitur diekstrak</div><div class="col-7">${data.feature_count || 70} fitur</div>`,
    `<div class="col-5 text-muted">Ukuran input</div><div class="col-7">${imgShape} px</div>`,
    `<div class="col-5 text-muted">ID Prediksi</div><div class="col-7 font-monospace">${escapeHtml((data.id || "-").slice(0, 8))}...</div>`,
    `<div class="col-5 text-muted">Waktu</div><div class="col-7">${formatTimestamp(data.timestamp)}</div>`,
    "</div></div>",
    '<div class="mt-3">',
    '<button class="btn btn-outline-secondary btn-sm w-100" type="button" data-action="reset-upload">',
    '<i class="bi bi-arrow-left me-1"></i>Prediksi Gambar Lain',
    "</button></div>",
  ].join("");
}

function renderScoreBars(scores) {
  const rows = [];
  if (scores.fertile != null) {
    const value = (scores.fertile * 100).toFixed(1);
    rows.push(scoreBar("Fertile", value, "success"));
  }
  if (scores.infertile != null) {
    const value = (scores.infertile * 100).toFixed(1);
    rows.push(scoreBar("Infertile", value, "danger"));
  }
  if (!rows.length) {
    return "";
  }
  return `<div class="mb-3"><div class="fw-semibold mb-2" style="font-size:.85rem">Distribusi Probabilitas Kelas</div>${rows.join("")}</div>`;
}

function scoreBar(label, value, colorClass) {
  return [
    '<div class="d-flex justify-content-between align-items-center mb-1" style="font-size:.8rem">',
    `<span class="text-${colorClass}">${label}</span><strong>${value}%</strong>`,
    "</div>",
    '<div class="progress mb-2" style="height:6px;border-radius:99px;background:#e2e8f0">',
    `<div class="progress-bar bg-${colorClass}" style="width:${value}%;border-radius:99px"></div>`,
    "</div>",
  ].join("");
}

function renderDistanceBars(distances, selectedClusterId, predictedLabel) {
  if (!Array.isArray(distances) || !distances.length) {
    return "";
  }

  const maxDistance = Math.max(...distances, 1e-6);
  const rows = distances.map((distance, index) => {
    const isSelected = index === selectedClusterId;
    const label = isSelected ? predictedLabel : predictedLabel === "fertile" ? "infertile" : "fertile";
    const color = label === "fertile" ? "#16a34a" : "#64748b";
    const badge = isSelected
      ? '<span class="badge ms-1" style="font-size:.65rem;background:' + color + ';color:#fff">dipilih</span>'
      : "";
    const width = ((distance / maxDistance) * 100).toFixed(1);
    return [
      '<div class="mb-2">',
      '<div class="d-flex justify-content-between align-items-center mb-1" style="font-size:.8rem">',
      `<span class="text-capitalize">${label}${badge}</span>`,
      `<span class="text-muted font-monospace">${Number(distance).toFixed(4)}</span>`,
      "</div>",
      '<div class="progress" style="height:7px;border-radius:99px;background:#e2e8f0">',
      `<div class="progress-bar" style="width:${width}%;border-radius:99px;background:${color};opacity:.7"></div>`,
      "</div></div>",
    ].join("");
  });

  return `<div class="mb-3"><div class="fw-semibold mb-2" style="font-size:.85rem">Jarak ke Pusat Cluster</div>${rows.join("")}<p class="text-muted mb-0" style="font-size:.73rem">Batang lebih pendek berarti sampel lebih dekat ke pusat cluster.</p></div>`;
}

function renderExplanationSection(explanation) {
  if (explanation.error) {
    return "";
  }

  const reasons = Array.isArray(explanation.reasons) ? explanation.reasons : [];
  const groups = Array.isArray(explanation.top_groups) ? explanation.top_groups : [];
  const groupBadges = groups.slice(0, 4).map((group) => {
    const pct = group.importance != null ? (group.importance * 100).toFixed(1) : "0.0";
    return `<span class="badge text-bg-light border me-1 mb-1">${escapeHtml(group.name)} ${pct}%</span>`;
  }).join("");

  return [
    '<div class="rounded-3 p-3 mb-3" style="background:#f8fafc;border:1px solid #e2e8f0">',
    '<div class="fw-semibold mb-2" style="font-size:.9rem">Penjelasan AWC</div>',
    explanation.summary ? `<p class="mb-2" style="font-size:.82rem">${escapeHtml(explanation.summary)}</p>` : "",
    reasons.length > 1 ? `<p class="text-muted mb-2" style="font-size:.78rem">${escapeHtml(reasons[1])}</p>` : "",
    groupBadges ? `<div>${groupBadges}</div>` : "",
    explanation.selected_feature_count ? `<p class="text-muted mb-0 mt-2" style="font-size:.75rem">AWC aktif memakai ${explanation.selected_feature_count} fitur terpilih untuk keputusan ini.</p>` : "",
    "</div>",
  ].join("");
}

function renderSegmentationSection(segmentation) {
  if (segmentation.error) {
    return `<div class="alert alert-warning py-2 px-3 mb-3" style="font-size:.8rem">Segmentasi U-Net belum tersedia: ${escapeHtml(segmentation.error)}</div>`;
  }
  if (!segmentation.overlay_b64) {
    return "";
  }

  const areas = segmentation.class_areas || {};
  const vascular = segmentation.vascular_metrics || {};
  const badges = [];
  Object.entries(areas).forEach(([name, value]) => {
    badges.push(`<span class="badge text-bg-light border me-1 mb-1">${escapeHtml(name)} ${Number(value).toFixed(2)}%</span>`);
  });

  const vascularCards = vascular.skeleton_length_norm != null ? [
    metricCard("Skeleton", vascular.skeleton_length_norm.toFixed(3), "Panjang total jaringan vaskular"),
    metricCard("Node", Number(vascular.node_count || 0).toFixed(0), "Jumlah titik percabangan"),
    metricCard("Density", Number(vascular.branch_density || 0).toFixed(4), "Kerapatan cabang terhadap skeleton"),
    metricCard("Area", Number(vascular.vascular_area_pct || 0).toFixed(2) + "%", "Persentase area vascularization"),
  ].join("") : "";

  return [
    '<div class="rounded-3 p-3 mb-3" style="background:#fff;border:1px solid #e2e8f0">',
    '<div class="fw-semibold mb-2" style="font-size:.9rem">Vascularization dan Mask U-Net</div>',
    '<div class="row g-2">',
    `<div class="col-md-4"><img src="${segmentation.original_b64}" class="img-fluid rounded-3 border" alt="Original"></div>`,
    `<div class="col-md-4"><img src="${segmentation.mask_b64}" class="img-fluid rounded-3 border" alt="Mask"></div>`,
    `<div class="col-md-4"><img src="${segmentation.overlay_b64}" class="img-fluid rounded-3 border" alt="Overlay"></div>`,
    "</div>",
    `<div class="mt-2">${badges.join("")}</div>`,
    vascularCards ? `<div class="row g-2 mt-1">${vascularCards}</div>` : "",
    "</div>",
  ].join("");
}

function metricCard(label, value, caption) {
  return [
    '<div class="col-sm-6 col-xl-3">',
    '<div class="rounded-3 p-2 h-100" style="background:#f8fafc;border:1px solid #e2e8f0">',
    `<div class="text-muted" style="font-size:.72rem">${label}</div>`,
    `<div class="fw-bold" style="font-size:1rem">${value}</div>`,
    `<div class="text-muted" style="font-size:.72rem">${caption}</div>`,
    "</div></div>",
  ].join("");
}

function renderComparisonTable(comparison) {
  const entries = Object.entries(comparison).filter(([, value]) => value && !value.error);
  if (!entries.length) {
    return "";
  }

  const rows = entries.map(([name, value]) => {
    const label = escapeHtml(value.label || "-");
    const confidence = value.confidence != null ? (Number(value.confidence) * 100).toFixed(1) + "%" : "-";
    return `<tr><td>${escapeHtml(name)}</td><td class="text-capitalize">${label}</td><td>${confidence}</td></tr>`;
  }).join("");

  return [
    '<div class="rounded-3 p-3 mb-3" style="background:#f8fafc;border:1px solid #e2e8f0">',
    '<div class="fw-semibold mb-2" style="font-size:.9rem">Perbandingan Prediksi Model</div>',
    '<div class="table-responsive">',
    '<table class="table table-sm align-middle mb-0">',
    '<thead><tr><th>Model</th><th>Prediksi</th><th>Confidence</th></tr></thead>',
    `<tbody>${rows}</tbody>`,
    "</table></div>",
    '<p class="text-muted mb-0 mt-2" style="font-size:.75rem">AWC tetap model utama. Tabel ini membantu melihat apakah sampel berada pada area yang ambigu antar model.</p>',
    "</div>",
  ].join("");
}

function confidenceTier(confidence) {
  if (confidence >= 0.9) {
    return { label: "Sangat Yakin", cls: "success", desc: "Model sangat yakin dengan prediksi ini." };
  }
  if (confidence >= 0.7) {
    return { label: "Cukup Yakin", cls: "warning", desc: "Model cukup yakin, tetapi tetap baik untuk diverifikasi lewat mask vaskular." };
  }
  return { label: "Kurang Yakin", cls: "danger", desc: "Prediksi berada pada area yang ambigu dan perlu ditinjau manual." };
}

function formatTimestamp(timestamp) {
  if (!timestamp) {
    return "-";
  }
  try {
    return new Date(timestamp).toLocaleString("id-ID");
  } catch (_error) {
    return timestamp;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initUploadPage);
} else {
  initUploadPage();
}
