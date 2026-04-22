/* ── analysis.js — Feature Importance + Cluster Visualisation ── */

const GROUP_COLORS = {
  'Statistik Intensitas': '#2563eb',
  'Histogram':            '#16a34a',
  'Tekstur LBP':          '#d97706',
  'Tepi (Edge)':          '#dc2626',
  'Lainnya':              '#64748b',
};

let vizChart = null;
let vizData  = null;

/* ─────────────────── Feature Importance ─────────────────── */
fetch('/api/analysis/feature-importance')
  .then(r => r.json())
  .then(d => {
    if (d.error) throw new Error(d.error);

    document.getElementById('fi-total-badge').textContent = d.total_features + ' fitur total';
    document.getElementById('fi-loading').classList.add('d-none');
    document.getElementById('fi-chart-wrap').classList.remove('d-none');
    document.getElementById('fg-loading').classList.add('d-none');
    document.getElementById('fg-chart-wrap').classList.remove('d-none');
    document.getElementById('fg-legend').classList.remove('d-none');

    /* Horizontal bar — top 15 */
    const top = d.top_features;
    const fiCtx = document.getElementById('fi-chart').getContext('2d');
    new Chart(fiCtx, {
      type: 'bar',
      data: {
        labels: top.map(f => f.name),
        datasets: [{
          label: 'Importance (%)',
          data: top.map(f => f.importance_pct),
          backgroundColor: top.map(f => (GROUP_COLORS[f.group] || '#64748b') + 'cc'),
          borderColor:     top.map(f =>  GROUP_COLORS[f.group] || '#64748b'),
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
              label: ctx => ' ' + ctx.parsed.x.toFixed(3) + '%',
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

    /* Donut — feature groups */
    const groupNames = Object.keys(d.groups);
    const groupVals  = groupNames.map(k => d.groups[k]);
    const fgCtx = document.getElementById('fg-chart').getContext('2d');
    new Chart(fgCtx, {
      type: 'doughnut',
      data: {
        labels: groupNames,
        datasets: [{
          data: groupVals,
          backgroundColor: groupNames.map(g => (GROUP_COLORS[g] || '#64748b') + 'dd'),
          borderColor: '#fff',
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        cutout: '62%',
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ' ' + ctx.parsed.toFixed(1) + '%' } },
        },
      },
    });

    /* Custom legend */
    const legendEl = document.getElementById('fg-legend');
    legendEl.innerHTML = groupNames.map((g, i) =>
      '<div class="d-flex align-items-center gap-2 mb-1">'
      + '<span class="legend-dot" style="background:' + (GROUP_COLORS[g] || '#64748b') + '"></span>'
      + '<span>' + g + '</span>'
      + '<span class="ms-auto fw-semibold">' + groupVals[i].toFixed(1) + '%</span>'
      + '</div>'
    ).join('');
  })
  .catch(err => {
    document.getElementById('fi-loading').innerHTML =
      '<i class="bi bi-exclamation-circle text-danger me-1"></i>Gagal memuat: ' + err.message;
    document.getElementById('fg-loading').innerHTML = '—';
  });

/* ─────────────────── Cluster Visualisation ──────────────── */
fetch('/api/analysis/cluster-visualization')
  .then(r => r.json())
  .then(d => {
    if (d.error) throw new Error(d.error);
    vizData = d;

    document.getElementById('viz-loading').classList.add('d-none');
    document.getElementById('viz-wrap').classList.remove('d-none');
    document.getElementById('viz-n-total').textContent    = d.n_samples;
    document.getElementById('viz-n-fertile').textContent  = d.n_fertile;
    document.getElementById('viz-n-infertile').textContent = d.n_infertile;

    renderViz('pca');
  })
  .catch(err => {
    document.getElementById('viz-loading').innerHTML =
      '<i class="bi bi-exclamation-circle text-danger me-1"></i>Gagal memuat: ' + err.message;
  });

/* Tab switching */
document.getElementById('viz-tabs').addEventListener('click', e => {
  const btn = e.target.closest('.tab-btn');
  if (!btn || !vizData) return;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderViz(btn.dataset.tab);
});

function renderViz(tab) {
  const isPca = tab === 'pca';
  const src   = isPca ? vizData.pca : vizData.tsne;

  if (isPca) {
    const v = vizData.pca.variance_explained;
    document.getElementById('viz-pca-info').textContent =
      'PC1 menjelaskan ' + v[0] + '% variansi · PC2 menjelaskan ' + v[1] + '% variansi · Total ' + (v[0]+v[1]).toFixed(1) + '%';
  } else {
    document.getElementById('viz-pca-info').textContent =
      't-SNE: proyeksi non-linear yang mempertahankan struktur lokal cluster (perplexity=30)';
  }

  const fertile   = src.points.filter(p => p.label === 'fertile');
  const infertile = src.points.filter(p => p.label === 'infertile');

  const datasets = [
    {
      label: 'Fertile',
      data: fertile.map(p => ({ x: p.x, y: p.y })),
      backgroundColor: 'rgba(22,163,74,.65)',
      borderColor:     '#16a34a',
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7,
    },
    {
      label: 'Infertile',
      data: infertile.map(p => ({ x: p.x, y: p.y })),
      backgroundColor: 'rgba(220,38,38,.65)',
      borderColor:     '#dc2626',
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7,
    },
  ];

  /* Centroids (PCA only) */
  if (isPca && vizData.pca.centroids) {
    datasets.push({
      label: 'Centroid',
      data: vizData.pca.centroids.map(c => ({ x: c.x, y: c.y, label: c.label })),
      backgroundColor: '#1e293b',
      borderColor:     '#fff',
      borderWidth: 2,
      pointRadius: 10,
      pointStyle: 'star',
      pointHoverRadius: 12,
    });
  }

  const xLabel = isPca ? 'PC1' : 't-SNE dim 1';
  const yLabel = isPca ? 'PC2' : 't-SNE dim 2';

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
            label: ctx => {
              const p = ctx.raw;
              if (ctx.dataset.label === 'Centroid') {
                return 'Centroid — ' + (p.label || '') + ' (' + p.x.toFixed(2) + ', ' + p.y.toFixed(2) + ')';
              }
              return ctx.dataset.label + ' (' + p.x.toFixed(3) + ', ' + p.y.toFixed(3) + ')';
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: xLabel, font: { size: 11 } }, grid: { color: '#f1f5f9' } },
        y: { title: { display: true, text: yLabel, font: { size: 11 } }, grid: { color: '#f1f5f9' } },
      },
    },
  });
}

/* ─────────────────── Confusion Matrix ──────────────────── */
fetch('/api/analysis/confusion-matrix')
  .then(r => r.json())
  .then(d => {
    if (d.error) throw new Error(d.error);

    document.getElementById('cm-samples-badge').textContent = d.test_samples + ' sampel test';

    const m = d.matrix;
    const pct = v => ((v / d.test_samples) * 100).toFixed(1);

    document.getElementById('cm-body').innerHTML =
      '<table class="table table-bordered text-center mb-3" style="font-size:.85rem">'
      + '<thead class="table-light"><tr>'
      + '<th class="text-muted" style="width:35%;border:none"></th>'
      + '<th colspan="2" class="text-muted" style="font-size:.78rem;font-weight:600">PREDIKSI MODEL</th>'
      + '</tr><tr>'
      + '<th class="text-muted" style="font-size:.78rem;font-weight:600">AKTUAL</th>'
      + '<th>Infertile</th><th>Fertile</th>'
      + '</tr></thead>'
      + '<tbody>'
      + '<tr><td class="text-start fw-semibold">Infertile</td>'
      + '<td style="background:#dcfce7;font-size:1.1rem;font-weight:700">' + m.tn
        + '<div style="font-size:.7rem;color:#15803d">TN · ' + pct(m.tn) + '%</div></td>'
      + '<td style="background:#fee2e2;font-size:1.1rem;font-weight:700">' + m.fp
        + '<div style="font-size:.7rem;color:#dc2626">FP · ' + pct(m.fp) + '%</div></td>'
      + '</tr>'
      + '<tr><td class="text-start fw-semibold">Fertile</td>'
      + '<td style="background:#fef9c3;font-size:1.1rem;font-weight:700">' + m.fn
        + '<div style="font-size:.7rem;color:#854d0e">FN · ' + pct(m.fn) + '%</div></td>'
      + '<td style="background:#dcfce7;font-size:1.1rem;font-weight:700">' + m.tp
        + '<div style="font-size:.7rem;color:#15803d">TP · ' + pct(m.tp) + '%</div></td>'
      + '</tr>'
      + '</tbody></table>'
      + '<div class="row g-2 text-center" style="font-size:.78rem">'
      + '<div class="col-6"><div style="background:#f0fdf4;border-radius:8px;padding:6px 4px"><div class="fw-bold text-success">' + (m.tn) + '</div><div class="text-muted">True Negative</div></div></div>'
      + '<div class="col-6"><div style="background:#fff1f2;border-radius:8px;padding:6px 4px"><div class="fw-bold text-danger">' + (m.fp) + '</div><div class="text-muted">False Positive</div></div></div>'
      + '<div class="col-6"><div style="background:#fefce8;border-radius:8px;padding:6px 4px"><div class="fw-bold text-warning">' + (m.fn) + '</div><div class="text-muted">False Negative</div></div></div>'
      + '<div class="col-6"><div style="background:#f0fdf4;border-radius:8px;padding:6px 4px"><div class="fw-bold text-success">' + (m.tp) + '</div><div class="text-muted">True Positive</div></div></div>'
      + '</div>';

    const met = d.metrics;
    const metPct = v => (v * 100).toFixed(1) + '%';
    const bar = (v, color) =>
      '<div class="progress mt-1" style="height:6px;border-radius:99px;background:#e2e8f0">'
      + '<div class="progress-bar" style="width:' + (v*100).toFixed(1) + '%;background:' + color + ';border-radius:99px"></div></div>';

    const metRows = [
      ['Akurasi',                   met.accuracy,    '#2563eb', 'Proporsi prediksi yang benar dari seluruh sampel'],
      ['Sensitivitas (Recall Fertile)', met.sensitivity, '#16a34a', 'Kemampuan mendeteksi telur fertile — tinggi = sedikit fertile yang terlewat'],
      ['Spesifisitas (Recall Infertile)', met.specificity, '#64748b', 'Kemampuan mendeteksi telur infertile — tinggi = sedikit infertile salah klasifikasi'],
      ['Presisi (PPV)',              met.precision,   '#0891b2', 'Dari yang diprediksi fertile, berapa yang benar-benar fertile'],
      ['NPV',                        met.npv,         '#7c3aed', 'Dari yang diprediksi infertile, berapa yang benar-benar infertile'],
      ['F1-Score',                   met.f1_score,    '#d97706', 'Harmonik mean antara presisi dan sensitivitas'],
    ];

    document.getElementById('metrics-body').innerHTML =
      '<div class="row g-2">'
      + metRows.map(([name, val, color, desc]) =>
          '<div class="col-sm-6">'
          + '<div class="p-2 rounded-3" style="background:#f8fafc;border:1px solid #e2e8f0">'
          + '<div class="d-flex justify-content-between align-items-center">'
          + '<span style="font-size:.8rem;font-weight:600">' + name + '</span>'
          + '<strong style="color:' + color + ';font-size:.95rem">' + metPct(val) + '</strong>'
          + '</div>'
          + bar(val, color)
          + '<div class="text-muted mt-1" style="font-size:.72rem">' + desc + '</div>'
          + '</div></div>'
        ).join('')
      + '</div>'
      + '<div class="mt-3 p-2 rounded-3" style="background:#fffbeb;border:1px solid #fde68a;font-size:.78rem">'
      + '<strong>Interpretasi:</strong> FP = telur infertile diprediksi fertile (ditetaskan sia-sia). '
      + 'FN = telur fertile diprediksi infertile (kehilangan embrio). '
      + 'Dalam konteks penetasan, FN lebih merugikan sehingga sensitivitas tinggi adalah prioritas.'
      + '</div>';
  })
  .catch(err => {
    document.getElementById('cm-body').innerHTML =
      '<p class="text-danger small mb-0"><i class="bi bi-exclamation-circle me-1"></i>' + err.message + '</p>';
    document.getElementById('metrics-body').innerHTML = '—';
  });
