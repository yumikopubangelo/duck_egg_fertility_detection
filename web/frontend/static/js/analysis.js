const GROUP_COLORS = {
  'Statistik Intensitas': '#2563eb',
  'Histogram': '#16a34a',
  'Tekstur LBP': '#d97706',
  'Tekstur GLCM': '#0891b2',
  'Morfologi Vaskular': '#7c3aed',
  'Tepi (Edge)': '#dc2626',
  'Lainnya': '#64748b',
};

const SEGMENT_COLORS = ['#1e293b', '#dc2626', '#16a34a'];

let vizChart = null;
let vizData = null;

const pct = (v) => (v * 100).toFixed(1) + '%';
const num = (v) => Number(v || 0).toLocaleString('id-ID');
const titleCase = (value) => String(value || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

function bar(value, color) {
  return (
    '<div class="progress mt-1" style="height:6px;border-radius:99px;background:#e2e8f0">' +
    '<div class="progress-bar" style="width:' + (value * 100).toFixed(1) + '%;background:' + color + ';border-radius:99px"></div>' +
    '</div>'
  );
}

fetch('/api/analysis/feature-importance')
  .then((r) => r.json())
  .then((d) => {
    if (d.error) throw new Error(d.error);

    document.getElementById('fi-total-badge').textContent = d.total_features + ' fitur total';
    document.getElementById('fi-loading').classList.add('d-none');
    document.getElementById('fi-chart-wrap').classList.remove('d-none');
    document.getElementById('fg-loading').classList.add('d-none');
    document.getElementById('fg-chart-wrap').classList.remove('d-none');
    document.getElementById('fg-legend').classList.remove('d-none');

    const top = d.top_features;
    const fiCtx = document.getElementById('fi-chart').getContext('2d');
    new Chart(fiCtx, {
      type: 'bar',
      data: {
        labels: top.map((f) => f.name),
        datasets: [{
          label: 'Importance (%)',
          data: top.map((f) => f.importance_pct),
          backgroundColor: top.map((f) => (GROUP_COLORS[f.group] || '#64748b') + 'cc'),
          borderColor: top.map((f) => GROUP_COLORS[f.group] || '#64748b'),
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ' ' + ctx.parsed.x.toFixed(3) + '%',
              afterLabel: (ctx) => {
                const f = top[ctx.dataIndex];
                return 'Kelompok: ' + f.group + '\nRank: #' + f.rank;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'Importance (%)', font: { size: 11 } },
            grid: { color: '#f1f5f9' },
          },
          y: { ticks: { font: { size: 11 }, color: '#334155' } },
        },
      },
    });

    const groupNames = Object.keys(d.groups);
    const groupVals = groupNames.map((k) => d.groups[k]);
    const fgCtx = document.getElementById('fg-chart').getContext('2d');
    new Chart(fgCtx, {
      type: 'doughnut',
      data: {
        labels: groupNames,
        datasets: [{
          data: groupVals,
          backgroundColor: groupNames.map((g) => (GROUP_COLORS[g] || '#64748b') + 'dd'),
          borderColor: '#fff',
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        cutout: '62%',
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => ' ' + ctx.parsed.toFixed(1) + '%' } },
        },
      },
    });

    const legendEl = document.getElementById('fg-legend');
    legendEl.innerHTML = groupNames.map((g, i) =>
      '<div class="d-flex align-items-center gap-2 mb-1">' +
      '<span class="legend-dot" style="background:' + (GROUP_COLORS[g] || '#64748b') + '"></span>' +
      '<span>' + g + '</span>' +
      '<span class="ms-auto fw-semibold">' + groupVals[i].toFixed(1) + '%</span>' +
      '</div>'
    ).join('');
  })
  .catch((err) => {
    document.getElementById('fi-loading').innerHTML =
      '<i class="bi bi-exclamation-circle text-danger me-1"></i>Gagal memuat: ' + err.message;
    document.getElementById('fg-loading').innerHTML = '-';
  });

fetch('/api/analysis/cluster-visualization')
  .then((r) => r.json())
  .then((d) => {
    if (d.error) throw new Error(d.error);
    vizData = d;

    document.getElementById('viz-loading').classList.add('d-none');
    document.getElementById('viz-wrap').classList.remove('d-none');
    document.getElementById('viz-n-total').textContent = d.n_samples;
    document.getElementById('viz-n-fertile').textContent = d.n_fertile;
    document.getElementById('viz-n-infertile').textContent = d.n_infertile;

    renderViz('pca');
  })
  .catch((err) => {
    document.getElementById('viz-loading').innerHTML =
      '<i class="bi bi-exclamation-circle text-danger me-1"></i>Gagal memuat: ' + err.message;
  });

document.getElementById('viz-tabs').addEventListener('click', (e) => {
  const btn = e.target.closest('.tab-btn');
  if (!btn || !vizData) return;
  document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
  btn.classList.add('active');
  renderViz(btn.dataset.tab);
});

function renderViz(tab) {
  const isPca = tab === 'pca';
  const src = isPca ? vizData.pca : vizData.tsne;

  if (isPca) {
    const v = vizData.pca.variance_explained;
    document.getElementById('viz-pca-info').textContent =
      'PC1 menjelaskan ' + v[0] + '% variansi, PC2 ' + v[1] + '%, total ' + (v[0] + v[1]).toFixed(1) + '%.';
  } else {
    document.getElementById('viz-pca-info').textContent =
      't-SNE mempertahankan struktur lokal cluster pada proyeksi dua dimensi.';
  }

  const fertile = src.points.filter((p) => p.label === 'fertile');
  const infertile = src.points.filter((p) => p.label === 'infertile');

  const datasets = [
    {
      label: 'Fertile',
      data: fertile.map((p) => ({ x: p.x, y: p.y })),
      backgroundColor: 'rgba(22,163,74,.65)',
      borderColor: '#16a34a',
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7,
    },
    {
      label: 'Infertile',
      data: infertile.map((p) => ({ x: p.x, y: p.y })),
      backgroundColor: 'rgba(220,38,38,.65)',
      borderColor: '#dc2626',
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7,
    },
  ];

  if (isPca && vizData.pca.centroids) {
    datasets.push({
      label: 'Centroid',
      data: vizData.pca.centroids.map((c) => ({ x: c.x, y: c.y, label: c.label })),
      backgroundColor: '#1e293b',
      borderColor: '#fff',
      borderWidth: 2,
      pointRadius: 10,
      pointStyle: 'star',
      pointHoverRadius: 12,
    });
  }

  if (vizChart) vizChart.destroy();
  const ctx = document.getElementById('viz-chart').getContext('2d');
  vizChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const p = ctx.raw;
              if (ctx.dataset.label === 'Centroid') {
                return 'Centroid - ' + (p.label || '') + ' (' + p.x.toFixed(2) + ', ' + p.y.toFixed(2) + ')';
              }
              return ctx.dataset.label + ' (' + p.x.toFixed(3) + ', ' + p.y.toFixed(3) + ')';
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: isPca ? 'PC1' : 't-SNE dim 1', font: { size: 11 } }, grid: { color: '#f1f5f9' } },
        y: { title: { display: true, text: isPca ? 'PC2' : 't-SNE dim 2', font: { size: 11 } }, grid: { color: '#f1f5f9' } },
      },
    },
  });
}

fetch('/api/analysis/confusion-matrix')
  .then((r) => r.json())
  .then((d) => {
    if (d.error) throw new Error(d.error);

    document.getElementById('cm-samples-badge').textContent = d.test_samples + ' sampel test';

    const m = d.matrix;
    const pctCell = (v) => ((v / d.test_samples) * 100).toFixed(1);

    document.getElementById('cm-body').innerHTML =
      '<table class="table table-bordered text-center mb-3" style="font-size:.85rem">' +
      '<thead class="table-light"><tr>' +
      '<th class="text-muted" style="width:35%;border:none"></th>' +
      '<th colspan="2" class="text-muted" style="font-size:.78rem;font-weight:600">PREDIKSI MODEL</th>' +
      '</tr><tr>' +
      '<th class="text-muted" style="font-size:.78rem;font-weight:600">AKTUAL</th>' +
      '<th>Infertile</th><th>Fertile</th>' +
      '</tr></thead>' +
      '<tbody>' +
      '<tr><td class="text-start fw-semibold">Infertile</td>' +
      '<td style="background:#dcfce7;font-size:1.1rem;font-weight:700">' + m.tn +
      '<div style="font-size:.7rem;color:#15803d">TN · ' + pctCell(m.tn) + '%</div></td>' +
      '<td style="background:#fee2e2;font-size:1.1rem;font-weight:700">' + m.fp +
      '<div style="font-size:.7rem;color:#dc2626">FP · ' + pctCell(m.fp) + '%</div></td>' +
      '</tr>' +
      '<tr><td class="text-start fw-semibold">Fertile</td>' +
      '<td style="background:#fef9c3;font-size:1.1rem;font-weight:700">' + m.fn +
      '<div style="font-size:.7rem;color:#854d0e">FN · ' + pctCell(m.fn) + '%</div></td>' +
      '<td style="background:#dcfce7;font-size:1.1rem;font-weight:700">' + m.tp +
      '<div style="font-size:.7rem;color:#15803d">TP · ' + pctCell(m.tp) + '%</div></td>' +
      '</tr>' +
      '</tbody></table>';

    const met = d.metrics;
    const metRows = [
      ['Akurasi', met.accuracy, '#2563eb', 'Proporsi prediksi yang benar dari seluruh sampel.'],
      ['Sensitivitas', met.sensitivity, '#16a34a', 'Kemampuan mendeteksi telur fertile.'],
      ['Spesifisitas', met.specificity, '#64748b', 'Kemampuan mendeteksi telur infertile.'],
      ['Presisi', met.precision, '#0891b2', 'Dari yang diprediksi fertile, berapa yang benar.'],
      ['NPV', met.npv, '#7c3aed', 'Dari yang diprediksi infertile, berapa yang benar.'],
      ['F1-Score', met.f1_score, '#d97706', 'Rata-rata harmonik presisi dan recall.'],
    ];

    document.getElementById('metrics-body').innerHTML =
      '<div class="row g-2">' +
      metRows.map(([name, val, color, desc]) =>
        '<div class="col-sm-6">' +
        '<div class="p-2 rounded-3" style="background:#f8fafc;border:1px solid #e2e8f0">' +
        '<div class="d-flex justify-content-between align-items-center">' +
        '<span style="font-size:.8rem;font-weight:600">' + name + '</span>' +
        '<strong style="color:' + color + ';font-size:.95rem">' + pct(val) + '</strong>' +
        '</div>' +
        bar(val, color) +
        '<div class="text-muted mt-1" style="font-size:.72rem">' + desc + '</div>' +
        '</div></div>'
      ).join('') +
      '</div>';
  })
  .catch((err) => {
    document.getElementById('cm-body').innerHTML =
      '<p class="text-danger small mb-0"><i class="bi bi-exclamation-circle me-1"></i>' + err.message + '</p>';
    document.getElementById('metrics-body').innerHTML = '-';
  });

fetch('/api/analysis/segmentation-report')
  .then((r) => r.json())
  .then((d) => {
    if (d.error) throw new Error(d.error);

    document.getElementById('seg-badge').textContent = d.dataset.test_samples + ' sampel test';
    document.getElementById('seg-loading').classList.add('d-none');
    document.getElementById('seg-wrap').classList.remove('d-none');
    document.getElementById('seg-checkpoint').textContent = d.checkpoint_path.split(/[\\/]/).pop();

    const summaryCards = [
      ['Mean Dice', d.summary.mean_dice, '#2563eb'],
      ['Mean IoU', d.summary.mean_iou, '#0891b2'],
      ['Foreground Dice', d.summary.foreground_mean_dice, '#16a34a'],
      ['Foreground IoU', d.summary.foreground_mean_iou, '#d97706'],
      ['Pixel Accuracy', d.summary.pixel_accuracy, '#7c3aed'],
      ['Total Pixel', d.summary.total_pixels, '#64748b', true],
    ];

    document.getElementById('seg-summary').innerHTML = summaryCards.map(([label, value, color, raw]) =>
      '<div class="col-sm-6 col-xl-2">' +
      '<div class="p-3 rounded-3 h-100" style="background:#f8fafc;border:1px solid #e2e8f0">' +
      '<div class="text-muted" style="font-size:.72rem">' + label + '</div>' +
      '<div class="fw-bold mt-1" style="font-size:1.1rem;color:' + color + '">' + (raw ? num(value) : pct(value)) + '</div>' +
      '</div></div>'
    ).join('');

    const classNames = d.classes.map((c) => titleCase(c.name));
    const totalPixels = d.summary.total_pixels || 1;
    document.getElementById('seg-cm-body').innerHTML =
      '<table class="table table-sm table-bordered text-center mb-0" style="font-size:.78rem">' +
      '<thead class="table-light"><tr><th>Aktual \\ Prediksi</th>' +
      classNames.map((name) => '<th>' + name + '</th>').join('') +
      '</tr></thead><tbody>' +
      d.confusion_matrix.map((row, rowIdx) =>
        '<tr><th class="text-start">' + classNames[rowIdx] + '</th>' +
        row.map((value, colIdx) => {
          const bg = rowIdx === colIdx ? '#dcfce7' : '#fff7ed';
          return '<td style="background:' + bg + '"><div class="fw-semibold">' + num(value) + '</div>' +
            '<div class="text-muted" style="font-size:.68rem">' + ((value / totalPixels) * 100).toFixed(2) + '%</div></td>';
        }).join('') +
        '</tr>'
      ).join('') +
      '</tbody></table>' +
      '<div class="text-muted mt-2" style="font-size:.72rem">Rows = ground truth, columns = prediction. Nilai diagonal menunjukkan pixel yang tersegmentasi benar.</div>';

    document.getElementById('seg-class-table').innerHTML =
      '<div class="table-responsive"><table class="table table-sm align-middle mb-0" style="font-size:.8rem">' +
      '<thead><tr><th>Kelas</th><th>Support</th><th>Dice</th><th>IoU</th><th>Recall</th><th>Precision</th></tr></thead><tbody>' +
      d.classes.map((cls, idx) =>
        '<tr>' +
        '<td><div class="d-flex align-items-center gap-2"><span class="legend-dot" style="background:' + (SEGMENT_COLORS[idx] || '#64748b') + '"></span>' + titleCase(cls.name) + '</div></td>' +
        '<td>' + num(cls.support_pixels) + '<div class="text-muted" style="font-size:.68rem">' + cls.support_pct.toFixed(2) + '%</div></td>' +
        '<td><strong>' + pct(cls.dice) + '</strong>' + bar(cls.dice, '#2563eb') + '</td>' +
        '<td><strong>' + pct(cls.iou) + '</strong>' + bar(cls.iou, '#16a34a') + '</td>' +
        '<td>' + pct(cls.recall) + '</td>' +
        '<td>' + pct(cls.precision) + '</td>' +
        '</tr>'
      ).join('') +
      '</tbody></table></div>';

    const exampleTitles = {
      best: 'Terbaik',
      median: 'Sedang',
      worst: 'Terburuk',
    };
    document.getElementById('seg-examples').innerHTML = Object.entries(exampleTitles).map(([key, label]) => {
      const ex = d.examples[key];
      if (!ex) return '';
      return (
        '<div class="col-xl-4 col-lg-6">' +
        '<div class="card h-100">' +
        '<div class="card-header d-flex justify-content-between align-items-center flex-wrap gap-2">' +
        '<span class="fw-semibold" style="font-size:.9rem">' + label + '</span>' +
        '<span class="text-muted" style="font-size:.72rem">' + ex.file_name + '</span>' +
        '</div>' +
        '<div class="card-body p-3">' +
        '<div class="d-flex gap-2 flex-wrap mb-2" style="font-size:.75rem">' +
        '<span class="badge text-bg-primary">Dice ' + pct(ex.mean_dice) + '</span>' +
        '<span class="badge text-bg-success">FG Dice ' + pct(ex.foreground_mean_dice) + '</span>' +
        '<span class="badge text-bg-warning">FG IoU ' + pct(ex.foreground_mean_iou) + '</span>' +
        '</div>' +
        '<div class="row g-2">' +
        '<div class="col-4"><img src="' + ex.original_b64 + '" class="img-fluid rounded border" alt="Original"><div class="text-muted mt-1" style="font-size:.68rem">Original</div></div>' +
        '<div class="col-4"><img src="' + ex.truth_b64 + '" class="img-fluid rounded border" alt="Ground truth"><div class="text-muted mt-1" style="font-size:.68rem">Ground truth</div></div>' +
        '<div class="col-4"><img src="' + ex.prediction_b64 + '" class="img-fluid rounded border" alt="Prediction"><div class="text-muted mt-1" style="font-size:.68rem">Prediksi</div></div>' +
        '</div>' +
        '</div></div></div>'
      );
    }).join('');
  })
  .catch((err) => {
    document.getElementById('seg-loading').innerHTML =
      '<i class="bi bi-exclamation-circle text-danger me-1"></i>Gagal memuat laporan segmentasi: ' + err.message;
  });
