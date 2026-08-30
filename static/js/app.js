// =========================================================
// CONFIG
// =========================================================
const MOCK_API = false; // set true to preview the UI without a running Flask backend
const VALIDATED_CONTAMINATION_RATE = 0.084; // <-- replace with your real value from cascade_evaluation.csv
const REQUIRED_COLUMNS = ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"];

// Columns that are prediction output, not raw sensor readings -- everything else in a
// row is treated as a sensor reading and shown dynamically in the detail panel.
const OUTPUT_COLUMNS = new Set([
  "Machine ID", "stage1_failure_probability", "stage1_flagged", "predicted_failure_types",
  "stage2_confidence", "shap_magnitude", "priority_score", "priority_rank", "top_features"
]);

const root = document.documentElement;

// =========================================================
// THEME (text/icon toggle, no sun/moon pairing -- just a moon icon that stays fixed,
// per the reference; if you'd rather it flip icons too, swap the class here)
// =========================================================
const themeBtn = document.getElementById("theme-toggle");
function applyTheme(theme) {
  root.setAttribute("data-theme", theme);
  themeBtn.textContent = theme === "dark" ? "Light" : "Dark";
  try { localStorage.setItem("pm-theme", theme); } catch (e) {}
}
let savedTheme = "dark";
try { savedTheme = localStorage.getItem("pm-theme") || "dark"; } catch (e) {}
applyTheme(savedTheme);
themeBtn.addEventListener("click", () => applyTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark"));

document.getElementById("banner-rate").textContent = (VALIDATED_CONTAMINATION_RATE * 100).toFixed(1) + "%";
document.getElementById("info-fp-rate").textContent = (VALIDATED_CONTAMINATION_RATE * 100).toFixed(1) + "%";
document.getElementById("required-cols-note").textContent = "Required columns: " + REQUIRED_COLUMNS.join(", ");

// =========================================================
// STATE
// =========================================================
let allResults = [];
let riskChart = null, typeChart = null;
let currentTopN = 10;
let currentSort = "rank";
let filters = { status: "all", type: "all", risk: "all", conf: "all", search: "" };

// =========================================================
// HELPERS
// =========================================================
function riskTier(prob) {
  const score = prob * 100;
  if (score >= 80) return "critical";
  if (score >= 50) return "monitor";
  return "healthy";
}
function confTier(conf) {
  const pct = conf * 100;
  if (pct >= 90) return "high";
  if (pct >= 70) return "medium";
  return "low";
}
function animateNumber(el, to, duration = 800) {
  const from = 0;
  const start = performance.now();
  function tick(now) {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(from + (to - from) * eased);
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// =========================================================
// UPLOAD: drag & drop + browse
// =========================================================
const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("file-input");

dropArea.addEventListener("click", () => fileInput.click());
["dragenter", "dragover"].forEach(evt =>
  dropArea.addEventListener(evt, e => { e.preventDefault(); dropArea.classList.add("drag-over"); }));
["dragleave", "drop"].forEach(evt =>
  dropArea.addEventListener(evt, e => { e.preventDefault(); dropArea.classList.remove("drag-over"); }));
dropArea.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

async function handleFile(file) {
  document.getElementById("upload-error").classList.add("hidden");

  // Client-side required-column check before sending anywhere
  const headerLine = await file.slice(0, 2000).text();
  const header = headerLine.split("\n")[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));
  const missing = REQUIRED_COLUMNS.filter(c => !header.includes(c));
  if (missing.length > 0) {
    document.getElementById("missing-cols-list").innerHTML = missing.map(c => `<li>${c}</li>`).join("");
    document.getElementById("upload-error").classList.remove("hidden");
    return;
  }

  runProcessingSequence(file);
}

function setStep(name, state) {
  const li = document.querySelector(`#processing-steps li[data-step="${name}"]`);
  li.className = `step-${state}`;
  li.textContent = (state === "done" ? "✓ " : state === "active" ? "◉ " : "○ ") + li.textContent.replace(/^[✓◉○]\s*/, "");
}

async function runProcessingSequence(file) {
  document.getElementById("upload-zone").classList.add("hidden");
  document.getElementById("processing-zone").classList.remove("hidden");
  document.getElementById("processing-file").textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;

  const steps = ["upload", "validate", "predict", "risk", "explain"];
  steps.forEach(s => setStep(s, "pending"));

  setStep("upload", "done");
  await sleep(300);
  setStep("validate", "done");
  await sleep(300);
  setStep("predict", "active");

  let results;
  try {
    results = MOCK_API ? await mockPredict(file) : await realPredict(file);
  } catch (err) {
    document.getElementById("processing-zone").classList.add("hidden");
    document.getElementById("upload-zone").classList.remove("hidden");
    alert("Prediction failed: " + err.message);
    return;
  }

  setStep("predict", "done");
  await sleep(200);
  setStep("risk", "done");
  await sleep(200);
  setStep("explain", "done");
  await sleep(400);

  document.getElementById("processing-zone").classList.add("hidden");
  document.getElementById("upload-zone").classList.remove("hidden");

  allResults = results;
  window.__lastUploadedFile = file;
  document.getElementById("last-run").textContent = "Last run · " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  populateTypeFilter(results);
  render(results);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function realPredict(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/predict", { method: "POST", body: formData });
  if (!res.ok) throw new Error(`Server responded ${res.status}`);
  return res.json();
}

// Mock data generator -- only used when MOCK_API = true, purely for previewing the UI
async function mockPredict(file) {
  await sleep(600);
  const n = 8 + Math.floor(Math.random() * 12);
  const types = ["PWF", "HDF", "TWF", "OSF", "RNF"];
  const arr = [];
  for (let i = 0; i < n; i++) {
    const prob = Math.random();
    const flagged = prob > 0.4;
    arr.push({
      "Machine ID": `M-${1000 + Math.floor(Math.random() * 9000)}`,
      "stage1_failure_probability": prob,
      "stage1_flagged": flagged ? 1 : 0,
      "predicted_failure_types": flagged ? types[Math.floor(Math.random() * types.length)] : "n/a (not flagged)",
      "stage2_confidence": flagged ? 0.7 + Math.random() * 0.3 : null,
      "shap_magnitude": flagged ? Math.random() * 5 : null,
      "priority_score": flagged ? Math.random() : null,
      "priority_rank": null,
      "top_features": flagged ? [
        { feature: "Tool wear [min]", value: Math.round(Math.random() * 200), impact: (Math.random() * 2 - 0.5) },
        { feature: "Torque [Nm]", value: Math.round(Math.random() * 80), impact: (Math.random() * 2 - 0.5) },
        { feature: "Process temperature [K]", value: (300 + Math.random() * 20).toFixed(1), impact: (Math.random() * 2 - 0.5) }
      ] : [],
      "Air temperature [K]": (295 + Math.random() * 10).toFixed(1),
      "Process temperature [K]": (305 + Math.random() * 10).toFixed(1),
      "Rotational speed [rpm]": Math.round(1200 + Math.random() * 800),
      "Torque [Nm]": (30 + Math.random() * 50).toFixed(1),
      "Tool wear [min]": Math.round(Math.random() * 220)
    });
  }
  arr.sort((a, b) => (b.priority_score || 0) - (a.priority_score || 0));
  arr.forEach((r, i) => r.priority_rank = i + 1);
  return arr;
}

// =========================================================
// RENDER
// =========================================================
function populateTypeFilter(results) {
  const select = document.getElementById("f-type");
  const types = [...new Set(results.map(r => r.predicted_failure_types).filter(t => t && t !== "n/a (not flagged)"))];
  select.innerHTML = `<option value="all">Failure type: All</option>` + types.map(t => `<option value="${t}">${t}</option>`).join("");
}

function render(results) {
  const scanned = results.length;
  const flagged = results.filter(r => r.stage1_flagged === 1).length;
  const critical = results.filter(r => riskTier(r.stage1_failure_probability || 0) === "critical").length;
  const monitor = results.filter(r => riskTier(r.stage1_failure_probability || 0) === "monitor").length;
  const healthy = results.filter(r => riskTier(r.stage1_failure_probability || 0) === "healthy").length;

  animateNumber(document.getElementById("kpi-scanned"), scanned);
  animateNumber(document.getElementById("kpi-flagged"), flagged);
  animateNumber(document.getElementById("kpi-critical"), critical);
  animateNumber(document.getElementById("kpi-monitor"), monitor);
  animateNumber(document.getElementById("kpi-healthy"), healthy);

  renderCharts(critical, monitor, healthy, results);
  applyFiltersAndRender();
}

function renderCharts(critical, monitor, healthy, results) {
  const style = getComputedStyle(root);
  const colors = { red: style.getPropertyValue("--red").trim(), amber: style.getPropertyValue("--amber").trim(), teal: style.getPropertyValue("--teal").trim() };
  const hasData = results.length > 0;

  document.getElementById("risk-empty").classList.toggle("hidden", hasData);
  document.getElementById("type-empty").classList.toggle("hidden", hasData);

  if (riskChart) riskChart.destroy();
  if (hasData) {
    riskChart = new Chart(document.getElementById("risk-chart"), {
      type: "bar",
      data: { labels: ["Critical", "Monitor", "Healthy"], datasets: [{ data: [critical, monitor, healthy], backgroundColor: [colors.red, colors.amber, colors.teal] }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => `${ctx.parsed.y} machines (${(ctx.parsed.y / results.length * 100).toFixed(0)}%)` } } },
        scales: { y: { beginAtZero: true, min: 0 } },
        animation: { duration: 900, easing: "easeOutQuart" }
      }
    });
  }

  const typeCounts = {};
  results.forEach(r => {
    const type = r.predicted_failure_types || "n/a";
    if (type === "n/a (not flagged)") return;
    typeCounts[type] = (typeCounts[type] || 0) + 1;
  });
  const palette = [colors.teal, colors.amber, colors.red, "#7F77DD", "#D4537E", "#5DCAA5"];

  if (typeChart) typeChart.destroy();
  if (hasData && Object.keys(typeCounts).length > 0) {
    typeChart = new Chart(document.getElementById("type-chart"), {
      type: "doughnut",
      data: { labels: Object.keys(typeCounts), datasets: [{ data: Object.values(typeCounts), backgroundColor: palette }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        cutout: "62%",
        plugins: { legend: { position: "right", labels: { color: style.getPropertyValue("--text-primary") } } },
        animation: { duration: 900, easing: "easeOutQuart" },
        onClick: (evt, elements) => {
          if (elements.length > 0) {
            const label = typeChart.data.labels[elements[0].index];
            document.getElementById("f-type").value = label;
            filters.type = label;
            applyFiltersAndRender();
          }
        }
      }
    });
  }
}

document.getElementById("chart-toggle").addEventListener("click", (e) => {
  const row = document.getElementById("charts-row");
  const hidden = row.classList.toggle("hidden");
  e.target.textContent = hidden ? "Show charts +" : "Hide charts −";
});

// ---- Filters, sort, top-N ----
["f-status", "f-type", "f-risk", "f-conf"].forEach(id => {
  document.getElementById(id).addEventListener("change", (e) => {
    const key = id.replace("f-", "");
    filters[key] = e.target.value;
    applyFiltersAndRender();
  });
});
let searchDebounce;
document.getElementById("f-search").addEventListener("input", (e) => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => { filters.search = e.target.value.trim().toLowerCase(); applyFiltersAndRender(); }, 250);
});
document.getElementById("sort-select").addEventListener("change", (e) => { currentSort = e.target.value; applyFiltersAndRender(); });
document.getElementById("top-n-select").addEventListener("change", (e) => {
  const customInput = document.getElementById("top-n-custom");
  if (e.target.value === "custom") {
    customInput.classList.remove("hidden");
    customInput.focus();
  } else {
    customInput.classList.add("hidden");
    currentTopN = parseInt(e.target.value, 10);
    applyFiltersAndRender();
  }
});
document.getElementById("top-n-custom").addEventListener("input", (e) => {
  const v = Math.max(1, Math.min(20, parseInt(e.target.value, 10) || 10));
  currentTopN = v;
  applyFiltersAndRender();
});

function applyFiltersAndRender() {
  let rows = allResults.filter(r => {
    const tier = riskTier(r.stage1_failure_probability || 0);
    if (filters.status !== "all" && tier !== filters.status) return false;
    if (filters.type !== "all" && r.predicted_failure_types !== filters.type) return false;
    if (filters.risk !== "all") {
      const score = (r.stage1_failure_probability || 0) * 100;
      const [lo, hi] = filters.risk.split("-").map(Number);
      if (score < lo || score > hi) return false;
    }
    if (filters.conf !== "all" && r.stage2_confidence != null && confTier(r.stage2_confidence) !== filters.conf) return false;
    if (filters.search && !String(r["Machine ID"] || "").toLowerCase().includes(filters.search)) return false;
    return true;
  });

  rows.sort((a, b) => {
    if (currentSort === "confidence") return (b.stage2_confidence || 0) - (a.stage2_confidence || 0);
    if (currentSort === "id") return String(a["Machine ID"]).localeCompare(String(b["Machine ID"]));
    if (currentSort === "type") return String(a.predicted_failure_types).localeCompare(String(b.predicted_failure_types));
    return (a.priority_rank ?? 999) - (b.priority_rank ?? 999);
  });

  const shown = rows.slice(0, currentTopN);
  renderTable(shown, rows.length);
}

function renderTable(shown, totalFiltered) {
  const tbody = document.getElementById("ranked-body");
  tbody.innerHTML = "";

  if (allResults.length === 0) {
    document.getElementById("table-caption").textContent = "No machines analyzed yet";
    tbody.innerHTML = `<tr><td colspan="6" class="text-center opacity-50 py-6 text-xs">No machines to show yet</td></tr>`;
    return;
  }
  if (shown.length === 0) {
    document.getElementById("table-caption").textContent = "No machines match your current filters";
    tbody.innerHTML = `<tr><td colspan="6" class="text-center opacity-50 py-6 text-xs">No machines match your current filters.</td></tr>`;
    return;
  }

  document.getElementById("table-caption").textContent = `Showing top ${shown.length} of ${totalFiltered} machines · sorted by ${currentSort === "rank" ? "highest risk" : currentSort}`;

  shown.forEach((r, i) => {
    const prob = r.stage1_failure_probability || 0;
    const tier = riskTier(prob);
    const score = Math.round(prob * 100);
    const conf = r.stage2_confidence != null ? Math.round(r.stage2_confidence * 100) : null;
    const statusLabel = tier === "critical" ? "Critical" : tier === "monitor" ? "Monitor" : "Stable";
    const barColor = tier === "critical" ? "var(--red)" : tier === "monitor" ? "var(--amber)" : "var(--teal)";

    const tr = document.createElement("tr");
    tr.className = `row-${tier} row-anim`;
    tr.style.animationDelay = `${i * 25}ms`;
    tr.innerHTML = `
      <td class="font-mono">${String(r.priority_rank ?? i + 1).padStart(2, "0")}</td>
      <td class="font-mono font-medium">${r["Machine ID"] ?? "--"}</td>
      <td><span class="badge">${(r.predicted_failure_types || "--").split(",")[0]}</span> <span class="status-${tier}">${statusLabel}</span></td>
      <td><span class="risk-bar-track"><span class="risk-bar-fill" style="width:0%;background:${barColor}"></span></span>${score}</td>
      <td class="font-mono">${conf != null ? conf + "%" : "--"}</td>
      <td class="opacity-50"><i class="ti ti-chevron-right"></i></td>
    `;
    tr.addEventListener("click", () => openDrawer(r));
    tbody.appendChild(tr);
    requestAnimationFrame(() => { tr.querySelector(".risk-bar-fill").style.width = score + "%"; });
  });
}

// =========================================================
// DETAIL DRAWER
// =========================================================
const drawer = document.getElementById("drawer");
const drawerOverlay = document.getElementById("drawer-overlay");

function recommendation(tier, type) {
  if (tier === "critical") return { title: "🔴 Immediate inspection", text: `Inspect the machine promptly. Predicted ${type} risk is high based on current sensor readings.` };
  if (tier === "monitor") return { title: "🟠 Schedule maintenance", text: `Schedule an inspection window for this machine. ${type} risk is elevated but not yet critical.` };
  return { title: "🟢 No immediate action", text: "Readings are within expected ranges. Continue routine monitoring." };
}

function openDrawer(r) {
  const prob = r.stage1_failure_probability || 0;
  const tier = riskTier(prob);
  const score = Math.round(prob * 100);
  const type = r.predicted_failure_types || "--";

  document.getElementById("drawer-id").textContent = r["Machine ID"] ?? "--";
  document.getElementById("drawer-summary").textContent = `Predicted ${type} · risk score ${score}`;

  const conf = r.stage2_confidence;
  const confPct = conf != null ? Math.round(conf * 100) : 0;
  document.getElementById("drawer-conf-val").textContent = conf != null ? confPct + "%" : "--";
  document.getElementById("drawer-conf-bar").style.width = "0%";
  requestAnimationFrame(() => { document.getElementById("drawer-conf-bar").style.width = confPct + "%"; });
  document.getElementById("drawer-conf-label").textContent = confPct >= 90 ? "High confidence prediction" : confPct >= 70 ? "Moderate confidence prediction" : conf != null ? "Low confidence prediction" : "Not flagged -- no classification made";

  // SHAP features
  const list = document.getElementById("drawer-features");
  const fullList = document.getElementById("drawer-shap-full");
  const expandBtn = document.getElementById("drawer-shap-expand");
  list.innerHTML = "";
  fullList.innerHTML = "";
  fullList.classList.add("hidden");
  expandBtn.classList.add("hidden");

  if (Array.isArray(r.top_features) && r.top_features.length > 0) {
    const maxAbs = Math.max(...r.top_features.map(f => Math.abs(f.impact)), 0.0001);
    r.top_features.forEach(f => {
      const pct = (Math.abs(f.impact) / maxAbs * 100).toFixed(0);
      const raises = f.impact >= 0;
      list.appendChild(shapRow(f.feature, f.value, pct, raises));
    });
    expandBtn.classList.remove("hidden");
    expandBtn.onclick = () => {
      fullList.classList.toggle("hidden");
      fullList.innerHTML = `<p class="text-xs opacity-60 mb-2">SHAP values indicate how each input contributed to the model's prediction relative to the fleet baseline.</p>` +
        r.top_features.map(f => `<div class="font-mono text-xs flex justify-between py-1"><span>${f.feature}</span><span style="color:${f.impact >= 0 ? "var(--red)" : "var(--teal)"}">${f.impact >= 0 ? "+" : ""}${f.impact.toFixed(2)}</span></div>`).join("");
    };
  } else {
    list.innerHTML = `<p class="text-xs opacity-60">Per-feature breakdown not available for this machine -- only an aggregate SHAP magnitude was returned: ${(r.shap_magnitude ?? 0).toFixed ? r.shap_magnitude.toFixed(2) : "--"}.</p>`;
  }

  // Recommendation
  const rec = recommendation(tier, type);
  document.getElementById("rec-title").textContent = rec.title;
  document.getElementById("rec-text").textContent = rec.text;

  // Sensor snapshot -- dynamically from whatever raw columns the row has
  const sensorGrid = document.getElementById("drawer-sensors");
  sensorGrid.innerHTML = "";
  Object.keys(r).forEach(key => {
    if (OUTPUT_COLUMNS.has(key)) return;
    const card = document.createElement("div");
    card.className = "sensor-card";
    card.innerHTML = `<p class="text-xs opacity-60">${key}</p><p class="font-mono text-sm">${r[key]}</p>`;
    sensorGrid.appendChild(card);
  });

  drawer.classList.remove("translate-x-full");
  drawerOverlay.classList.remove("hidden");
}

function shapRow(feature, value, pct, raises) {
  const div = document.createElement("div");
  div.className = "shap-row";
  div.innerHTML = `
    <div class="flex justify-between text-xs mb-1">
      <span class="font-medium">${feature}</span>
      <span class="opacity-60 font-mono">${value}</span>
    </div>
    <div class="shap-track"><div class="shap-fill" style="width:0%;background:${raises ? "var(--red)" : "var(--teal)"}"></div></div>
    <p class="text-xs mt-1" style="color:${raises ? "var(--red)" : "var(--teal)"}">${raises ? "Raises risk" : "Lowers risk"}</p>
  `;
  requestAnimationFrame(() => { div.querySelector(".shap-fill").style.width = pct + "%"; });
  return div;
}

function closeDrawer() { drawer.classList.add("translate-x-full"); drawerOverlay.classList.add("hidden"); }
document.getElementById("drawer-close").addEventListener("click", closeDrawer);
drawerOverlay.addEventListener("click", closeDrawer);

// =========================================================
// DATASET PREVIEW MODAL
// =========================================================
document.getElementById("view-dataset-btn").addEventListener("click", async () => {
  const file = window.__lastUploadedFile;
  if (!file) { alert("Upload a CSV first."); return; }
  const text = await file.text();
  const lines = text.split("\n").filter(l => l.trim().length > 0);
  const header = lines[0].split(",");
  const dataRows = lines.slice(1);
  let missing = 0;
  dataRows.forEach(line => { line.split(",").forEach(cell => { if (cell.trim() === "") missing++; }); });

  document.getElementById("ds-rows").textContent = dataRows.length;
  document.getElementById("ds-cols").textContent = header.length;
  document.getElementById("ds-missing").textContent = missing;
  document.getElementById("ds-machines").textContent = allResults.length || "--";

  const preview = dataRows.slice(0, 10).map(l => l.split(","));
  const table = document.getElementById("ds-table");
  table.innerHTML = "<thead><tr>" + header.map(h => `<th class="text-left opacity-60 pr-3 pb-1">${h}</th>`).join("") + "</tr></thead>" +
    "<tbody>" + preview.map(row => "<tr>" + row.map(c => `<td class="pr-3 py-1 font-mono">${c}</td>`).join("") + "</tr>").join("") + "</tbody>";

  document.getElementById("dataset-modal").classList.remove("hidden");
  document.getElementById("dataset-modal-overlay").classList.remove("hidden");
});
document.getElementById("dataset-modal-close").addEventListener("click", closeDatasetModal);
document.getElementById("dataset-modal-overlay").addEventListener("click", closeDatasetModal);
function closeDatasetModal() {
  document.getElementById("dataset-modal").classList.add("hidden");
  document.getElementById("dataset-modal-overlay").classList.add("hidden");
}

// =========================================================
// MODEL INFO MODAL
// =========================================================
document.getElementById("info-btn").addEventListener("click", () => {
  document.getElementById("info-modal").classList.remove("hidden");
  document.getElementById("info-modal-overlay").classList.remove("hidden");
});
document.getElementById("info-modal-close").addEventListener("click", closeInfoModal);
document.getElementById("info-modal-overlay").addEventListener("click", closeInfoModal);
function closeInfoModal() {
  document.getElementById("info-modal").classList.add("hidden");
  document.getElementById("info-modal-overlay").classList.add("hidden");
}

// =========================================================
// DOWNLOAD RANKING CSV (client-side, from data already in memory)
// =========================================================
document.getElementById("download-btn").addEventListener("click", () => {
  if (allResults.length === 0) { alert("No results to download yet."); return; }
  const cols = ["Machine ID", "predicted_failure_types", "stage1_failure_probability", "stage2_confidence", "priority_rank"];
  const header = ["Machine ID", "Predicted failure", "Risk score", "Confidence", "Rank"];
  const lines = [header.join(",")];
  allResults.forEach(r => {
    lines.push([
      r["Machine ID"], r.predicted_failure_types,
      Math.round((r.stage1_failure_probability || 0) * 100),
      r.stage2_confidence != null ? Math.round(r.stage2_confidence * 100) + "%" : "",
      r.priority_rank ?? ""
    ].join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "ranking.csv"; a.click();
  URL.revokeObjectURL(url);
});

// =========================================================
// INITIAL EMPTY STATE
// =========================================================
render([]);