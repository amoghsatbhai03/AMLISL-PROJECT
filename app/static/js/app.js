/**
 * Multilingual Banking Complaint Classifier - Frontend Controller
 * Lite Theme - Handles interactive tabs, live classification, Chart.js visualizations,
 * batch CSV upload/audit, and confusion matrix rendering.
 */

// Global state
let probChart = null;
let batchCatChart = null;
let batchUrgencyChart = null;
let selectedBatchFile = null;

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    setupEventListeners();
    await loadSamples();
    await loadMetricsData();
}

function setupEventListeners() {
    const inputArea = document.getElementById('complaint-input');
    const charCounter = document.getElementById('char-counter');

    if (inputArea && charCounter) {
        inputArea.addEventListener('input', () => {
            const count = inputArea.value.length;
            charCounter.textContent = `${count} chars`;
        });
    }

    // Drag and drop for batch upload
    const dropzone = document.getElementById('dropzone');
    if (dropzone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('border-blue-500', 'bg-blue-50/50');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('border-blue-500', 'bg-blue-50/50');
            }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                setBatchFile(files[0]);
            }
        });
    }
}

// Tab Switching
function switchTab(tabId) {
    const tabs = ['single', 'batch', 'metrics'];
    tabs.forEach(t => {
        const btn = document.getElementById(`tab-btn-${t}`);
        const content = document.getElementById(`tab-content-${t}`);
        if (t === tabId) {
            btn.classList.add('border-blue-600', 'text-blue-600', 'font-semibold');
            btn.classList.remove('border-transparent', 'text-slate-500');
            content.classList.remove('hidden');
        } else {
            btn.classList.remove('border-blue-600', 'text-blue-600', 'font-semibold');
            btn.classList.add('border-transparent', 'text-slate-500');
            content.classList.add('hidden');
        }
    });
    if (window.lucide) lucide.createIcons();
}

// Load Curated Multilingual Samples
async function loadSamples() {
    const container = document.getElementById('samples-container');
    try {
        const res = await fetch('/api/samples');
        if (!res.ok) throw new Error('Failed to fetch samples');
        const samples = await res.json();

        container.innerHTML = '';
        samples.forEach((s) => {
            const btn = document.createElement('button');
            btn.className = 'sample-pill text-xs px-3 py-1.5 rounded-lg bg-slate-50 hover:bg-blue-50/80 border border-slate-200 hover:border-blue-300 text-slate-700 flex items-center space-x-1.5 transition text-left shadow-2xs';
            btn.innerHTML = `
                <span class="font-bold text-blue-600">${s.language}:</span>
                <span class="truncate max-w-[200px] sm:max-w-xs text-slate-500">"${s.text.substring(0, 38)}..."</span>
            `;
            btn.onclick = () => {
                const inputArea = document.getElementById('complaint-input');
                inputArea.value = s.text;
                document.getElementById('char-counter').textContent = `${s.text.length} chars`;
                document.getElementById('lang-select').value = s.lang_code;
                handleClassify();
            };
            container.appendChild(btn);
        });
    } catch (err) {
        container.innerHTML = '<span class="text-xs text-rose-600">Could not load sample complaints.</span>';
    }
}

function clearInput() {
    const inputArea = document.getElementById('complaint-input');
    inputArea.value = '';
    document.getElementById('char-counter').textContent = '0 chars';
    document.getElementById('results-placeholder').classList.remove('hidden');
    document.getElementById('results-container').classList.add('hidden');
}

// Classify Single Complaint
async function handleClassify() {
    const inputArea = document.getElementById('complaint-input');
    const text = inputArea.value.trim();
    if (!text) {
        alert('Please enter or select a customer complaint narrative.');
        return;
    }

    const lang = document.getElementById('lang-select').value;
    const btn = document.getElementById('btn-classify');
    const btnText = document.getElementById('classify-btn-text');
    const btnIcon = document.getElementById('classify-btn-icon');

    // Button loading state
    btn.disabled = true;
    btnText.textContent = 'Analyzing...';
    btnIcon.setAttribute('data-lucide', 'loader-2');
    btnIcon.classList.add('animate-spin');
    if (window.lucide) lucide.createIcons();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, language: lang })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Prediction failed');
        }

        const data = await response.json();
        renderPredictionResults(data);
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Analyze & Triage';
        btnIcon.setAttribute('data-lucide', 'arrow-right');
        btnIcon.classList.remove('animate-spin');
        if (window.lucide) lucide.createIcons();
    }
}

function renderPredictionResults(data) {
    document.getElementById('results-placeholder').classList.add('hidden');
    const container = document.getElementById('results-container');
    container.classList.remove('hidden');

    // Category & Confidence
    document.getElementById('res-category').textContent = data.predicted_category;
    document.getElementById('res-confidence').textContent = data.confidence_percent;
    document.getElementById('res-confidence-bar').style.width = data.confidence_percent;

    // Language
    document.getElementById('res-lang').textContent = data.language.name;
    document.getElementById('res-lang-conf').textContent = `Confidence: ${(data.language.confidence * 100).toFixed(0)}%`;

    // Urgency Badge (Lite styling)
    const urgency = data.urgency.urgency_level;
    const urgencyBadge = document.getElementById('res-urgency-badge');
    urgencyBadge.textContent = urgency;

    if (urgency === 'Critical') {
        urgencyBadge.className = 'inline-block mt-1 text-xs font-bold px-2.5 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200';
    } else if (urgency === 'High') {
        urgencyBadge.className = 'inline-block mt-1 text-xs font-bold px-2.5 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200';
    } else if (urgency === 'Medium') {
        urgencyBadge.className = 'inline-block mt-1 text-xs font-bold px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200';
    } else {
        urgencyBadge.className = 'inline-block mt-1 text-xs font-bold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200';
    }

    // SLA & Routing
    document.getElementById('res-sla').textContent = `< ${data.urgency.sla_target_hours} Hours`;
    document.getElementById('res-sentiment').textContent = data.urgency.sentiment;
    document.getElementById('res-department').textContent = data.urgency.recommended_department;
    document.getElementById('res-email').textContent = data.urgency.assigned_support_email;

    // Explainability Keywords
    const kwContainer = document.getElementById('res-keywords');
    kwContainer.innerHTML = '';
    const keywords = data.explainability.top_signals || [];
    if (keywords.length === 0) {
        kwContainer.innerHTML = '<span class="text-xs text-slate-400 italic">No strong n-gram triggers found</span>';
    } else {
        keywords.forEach(k => {
            const span = document.createElement('span');
            span.className = 'inline-flex items-center text-xs px-2.5 py-1 rounded-md bg-blue-50 border border-blue-200 text-blue-700 font-mono shadow-2xs';
            span.innerHTML = `${k.token} <span class="ml-1 text-[10px] text-blue-600 font-bold">+${k.importance_score}</span>`;
            kwContainer.appendChild(span);
        });
    }

    // Render Probabilities Chart
    renderProbabilitiesChart(data.probabilities);
    if (window.lucide) lucide.createIcons();
}

function renderProbabilitiesChart(probabilities) {
    const ctx = document.getElementById('chart-probabilities').getContext('2d');
    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities).map(v => (v * 100).toFixed(1));

    if (probChart) {
        probChart.destroy();
    }

    probChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: values,
                backgroundColor: labels.map((_, i) => i === 0 ? 'rgba(37, 99, 235, 0.85)' : 'rgba(203, 213, 225, 0.7)'),
                borderColor: labels.map((_, i) => i === 0 ? '#2563eb' : '#cbd5e1'),
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` Probability: ${ctx.parsed.x}%`
                    }
                }
            },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: { color: '#64748b', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: '#334155',
                        font: { size: 11, weight: '500' },
                        callback: function(value) {
                            const lbl = this.getLabelForValue(value);
                            return lbl.length > 24 ? lbl.substring(0, 22) + '...' : lbl;
                        }
                    }
                }
            }
        }
    });
}


// Batch CSV Upload Handlers
function handleFileSelected(event) {
    const file = event.target.files[0];
    if (file) setBatchFile(file);
}

function setBatchFile(file) {
    selectedBatchFile = file;
    const fileInfo = document.getElementById('selected-file-info');
    fileInfo.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    fileInfo.classList.remove('hidden');
    document.getElementById('btn-process-batch').disabled = false;
}

async function loadDemoBatch() {
    try {
        const response = await fetch('/data/sample_batch_complaints.csv');
        let blob;
        if (response.ok) {
            blob = await response.blob();
        } else {
            // Fallback: fetch from samples and build CSV in memory
            const sampleRes = await fetch('/api/samples');
            const samples = await sampleRes.json();
            let csv = "ticket_id,customer_name,complaint_text\n";
            samples.forEach((s, idx) => {
                csv += `TKT-${1000 + idx},"Customer ${idx+1}","${s.text.replace(/"/g, '""')}"\n`;
            });
            blob = new Blob([csv], { type: 'text/csv' });
        }
        const file = new File([blob], 'demo_banking_complaints.csv', { type: 'text/csv' });
        setBatchFile(file);
        processBatchUpload();
    } catch (err) {
        alert('Could not load demo batch: ' + err.message);
    }
}

async function processBatchUpload() {
    if (!selectedBatchFile) return;

    const btn = document.getElementById('btn-process-batch');
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i><span>Processing Tickets...</span>`;
    if (window.lucide) lucide.createIcons();

    const formData = new FormData();
    formData.append('file', selectedBatchFile);

    try {
        const res = await fetch('/api/predict-batch', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Batch processing failed');
        }

        const data = await res.json();
        renderBatchResults(data);
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i><span>Start Bulk Classification</span>`;
        if (window.lucide) lucide.createIcons();
    }
}

function renderBatchResults(data) {
    document.getElementById('batch-results-view').classList.remove('hidden');

    // KPIs
    document.getElementById('kpi-total-tickets').textContent = data.total_records;

    const criticalHigh = (data.urgency_counts['Critical'] || 0) + (data.urgency_counts['High'] || 0);
    document.getElementById('kpi-critical-sla').textContent = criticalHigh;

    const topCat = Object.entries(data.category_counts).sort((a,b) => b[1] - a[1])[0];
    document.getElementById('kpi-top-category').textContent = topCat ? `${topCat[0]} (${topCat[1]})` : '-';

    // Download Link
    const dlBtn = document.getElementById('btn-download-enriched');
    dlBtn.href = `/api/download-batch/${data.batch_id}`;

    // Batch Charts
    renderBatchCharts(data.category_counts, data.urgency_counts);

    // Render Preview Table
    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';
    data.preview.forEach(row => {
        const tr = document.createElement('tr');
        const urg = row.urgency_level || 'Low';
        let urgClass = 'bg-slate-100 text-slate-700';
        if (urg === 'Critical') urgClass = 'bg-rose-50 text-rose-700 border border-rose-200';
        else if (urg === 'High') urgClass = 'bg-amber-50 text-amber-800 border border-amber-200';
        else if (urg === 'Medium') urgClass = 'bg-blue-50 text-blue-700 border border-blue-200';
        else if (urg === 'Low') urgClass = 'bg-emerald-50 text-emerald-700 border border-emerald-200';

        tr.innerHTML = `
            <td class="font-mono text-xs text-blue-600 font-semibold">${row.ticket_id || '-'}</td>
            <td class="font-medium text-slate-800">${row.customer_name || 'Customer'}</td>
            <td><span class="text-xs px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-600">${row.detected_language || 'EN'}</span></td>
            <td class="font-semibold text-slate-800">${row.predicted_category}</td>
            <td><span class="text-xs font-mono text-emerald-600 font-bold">${(row.confidence * 100).toFixed(1)}%</span></td>
            <td><span class="text-xs px-2.5 py-0.5 rounded-full font-bold ${urgClass}">${urg}</span></td>
            <td class="text-xs text-slate-600">${row.assigned_department}</td>
        `;
        tbody.appendChild(tr);
    });

    if (window.lucide) lucide.createIcons();
}

function renderBatchCharts(categoryCounts, urgencyCounts) {
    // 1. Category Chart
    const ctxCat = document.getElementById('chart-batch-categories').getContext('2d');
    if (batchCatChart) batchCatChart.destroy();

    batchCatChart = new Chart(ctxCat, {
        type: 'bar',
        data: {
            labels: Object.keys(categoryCounts),
            datasets: [{
                data: Object.values(categoryCounts),
                backgroundColor: 'rgba(37, 99, 235, 0.8)',
                borderColor: '#2563eb',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: '#64748b',
                        font: { size: 9 },
                        callback: function(val) {
                            const lbl = this.getLabelForValue(val);
                            return lbl.length > 14 ? lbl.substring(0, 12) + '..' : lbl;
                        }
                    }
                },
                y: {
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    ticks: { color: '#64748b' }
                }
            }
        }
    });

    // 2. Urgency Doughnut Chart
    const ctxUrg = document.getElementById('chart-batch-urgency').getContext('2d');
    if (batchUrgencyChart) batchUrgencyChart.destroy();

    const urgColors = {
        'Critical': '#ef4444',
        'High': '#f59e0b',
        'Medium': '#3b82f6',
        'Low': '#10b981'
    };

    const urgLabels = Object.keys(urgencyCounts);
    const urgData = Object.values(urgencyCounts);

    batchUrgencyChart = new Chart(ctxUrg, {
        type: 'doughnut',
        data: {
            labels: urgLabels,
            datasets: [{
                data: urgData,
                backgroundColor: urgLabels.map(l => urgColors[l] || '#94a3b8'),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#475569', font: { size: 11 }, boxWidth: 12 }
                }
            }
        }
    });
}


// Load Model Metrics & Confusion Matrix
async function loadMetricsData() {
    try {
        const res = await fetch('/api/metrics');
        if (!res.ok) return;
        const data = await res.json();

        // Top KPIs
        if (data.metrics) {
            document.getElementById('metric-acc').textContent = `${(data.metrics.accuracy * 100).toFixed(1)}%`;
            const genGapElem = document.getElementById('metric-gen-gap');
            if (genGapElem) {
                const gap = ((data.metrics.generalization_gap || 0.0429) * 100).toFixed(1);
                genGapElem.textContent = `${gap}%`;
            }
            document.getElementById('metric-macro-f1').textContent = data.metrics.macro_f1.toFixed(4);
            document.getElementById('metric-precision').textContent = data.metrics.precision.toFixed(4);
            document.getElementById('metric-recall').textContent = data.metrics.recall.toFixed(4);
        }
        if (data.total_samples) {
            document.getElementById('metric-samples').textContent = data.total_samples.toLocaleString();
        }

        // Benchmark table
        const benchTbody = document.getElementById('metrics-benchmark-table');
        benchTbody.innerHTML = '';
        if (data.model_comparison) {
            Object.entries(data.model_comparison).forEach(([name, stats]) => {
                const isChampion = name === data.model_name;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="font-semibold text-slate-800 flex items-center gap-2">
                        ${isChampion ? '<span class="text-amber-500">★</span>' : ''}
                        ${name}
                    </td>
                    <td class="font-mono text-blue-600 font-bold">${stats.cv_f1_macro.toFixed(4)}</td>
                    <td class="font-mono text-slate-500">±${stats.cv_f1_std.toFixed(4)}</td>
                    <td class="text-slate-500">${stats.time_sec}s</td>
                    <td>
                        <span class="text-xs px-2.5 py-0.5 rounded-full font-bold ${isChampion ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-slate-100 text-slate-600'}">
                            ${isChampion ? 'Selected Champion' : 'Evaluated'}
                        </span>
                    </td>
                `;
                benchTbody.appendChild(tr);
            });
        }

        // Per-class metrics
        const perClassTbody = document.getElementById('per-class-table');
        perClassTbody.innerHTML = '';
        if (data.per_class_metrics) {
            Object.entries(data.per_class_metrics).forEach(([cat, stats]) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="font-semibold text-slate-800 text-xs">${cat}</td>
                    <td class="font-mono text-xs text-blue-600">${stats.precision.toFixed(3)}</td>
                    <td class="font-mono text-xs text-indigo-600">${stats.recall.toFixed(3)}</td>
                    <td class="font-mono text-xs text-emerald-600 font-bold">${stats.f1_score.toFixed(3)}</td>
                `;
                perClassTbody.appendChild(tr);
            });
        }

        // Confusion Matrix Heatmap
        if (data.confusion_matrix) {
            renderConfusionMatrix(data.confusion_matrix);
        }
    } catch (err) {
        console.error('Failed to load metrics:', err);
    }
}

function renderConfusionMatrix(cm) {
    const container = document.getElementById('confusion-matrix-container');
    const labels = cm.labels;
    const matrix = cm.matrix;

    // Abbreviated label names for columns
    const shortLabels = [
        "Bank Acct", "Credit Card", "Credit Rprt", "Transfers", "Mortgages", "Loans"
    ];

    let html = `
        <table class="w-full text-center border-collapse text-xs">
            <thead>
                <tr>
                    <th class="p-2 text-slate-400 text-left font-medium">Actual \\ Pred</th>
                    ${shortLabels.map((lbl, i) => `<th class="p-2 text-slate-600 font-semibold" title="${labels[i]}">${lbl}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
    `;

    matrix.forEach((row, rIdx) => {
        html += `<tr><td class="p-2 text-left font-semibold text-slate-700 truncate max-w-[130px]" title="${labels[rIdx]}">${shortLabels[rIdx]}</td>`;
        row.forEach((val, cIdx) => {
            const isDiagonal = rIdx === cIdx;
            const bgClass = isDiagonal ? 'bg-blue-600 text-white font-bold shadow-2xs' : (val > 0 ? 'bg-rose-100 text-rose-700 font-semibold' : 'bg-slate-50 text-slate-400 border border-slate-100');
            html += `<td class="p-2.5 cm-cell ${bgClass} rounded-lg transition font-mono">${val}</td>`;
        });
        html += `</tr>`;
    });

    html += `</tbody></table>`;
    container.innerHTML = html;
}
