const FINAL_STATUSES = new Set(["completed", "partial", "failed", "canceled"]);

const state = {
  apiBase: "",
  clientId: "vps-console",
  uploadId: null,
  uploadedInputs: [],
  currentJobId: null,
  currentJobStatus: null,
  pollTimer: null,
};

function el(id) {
  return document.getElementById(id);
}

function normalizeApiBase(value) {
  const raw = (value || "").trim();
  if (!raw) return "";
  return raw.replace(/\/+$/, "");
}

function apiUrl(path) {
  return `${state.apiBase}${path}`;
}

async function apiFetch(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  if (!response.ok) {
    const detail = data?.detail || data?.error || `${response.status} ${response.statusText}`;
    throw new Error(detail);
  }
  return data;
}

function statusClass(status) {
  return `status ${String(status || "").toLowerCase()}`;
}

function shortPath(path) {
  if (!path) return "-";
  const chunks = String(path).split(/[\\/]/);
  return chunks[chunks.length - 1] || path;
}

function isoNow() {
  return new Date().toISOString();
}

function newRequestId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

function setReadyBadge(kind, text) {
  const badge = el("readyBadge");
  badge.textContent = text;
  badge.style.background =
    kind === "ok" ? "#def8ea" : kind === "warn" ? "#fff2dc" : "#ffe3de";
  badge.style.color = kind === "ok" ? "#155c38" : kind === "warn" ? "#825b10" : "#8b2d22";
}

function setJobBadge(text) {
  el("jobBadge").textContent = text;
}

function appendEventsLog(events) {
  const lines = events.map((event) => {
    const ts = event.timestamp ? new Date(event.timestamp).toLocaleString() : isoNow();
    const filePart = event.file_id ? ` file=${shortPath(event.file_id)}` : "";
    const payloadText = event.payload ? ` ${JSON.stringify(event.payload)}` : "";
    return `[${ts}] ${event.event_code}${filePart}${payloadText}`;
  });
  el("eventsLog").textContent = lines.join("\n");
}

function renderSelectedFiles(files) {
  const list = el("selectedList");
  const count = el("selectedCount");
  list.innerHTML = "";
  if (!files.length) {
    count.textContent = "No files selected";
    return;
  }
  count.textContent = `${files.length} file(s) selected`;
  for (const file of files) {
    const li = document.createElement("li");
    li.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
    list.appendChild(li);
  }
}

function renderJobsTable(jobs) {
  const tbody = el("jobsTbody");
  tbody.innerHTML = "";
  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="4">No jobs yet</td></tr>';
    return;
  }
  for (const job of jobs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${job.job_id}</td>
      <td><span class="${statusClass(job.status)}">${job.status}</span></td>
      <td>${job.mode}</td>
      <td><button class="btn ghost btn-monitor" data-job-id="${job.job_id}">Monitor</button></td>
    `;
    tbody.appendChild(tr);
  }
}

function renderFilesTable(files) {
  const tbody = el("filesTbody");
  tbody.innerHTML = "";
  if (!files.length) {
    tbody.innerHTML = '<tr><td colspan="4">No files tracked yet</td></tr>';
    el("filesSummary").textContent = "-";
    return;
  }

  const counts = {};
  for (const file of files) {
    counts[file.status] = (counts[file.status] || 0) + 1;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td title="${file.input_path}">${shortPath(file.input_path)}</td>
      <td><span class="${statusClass(file.status)}">${file.status}</span></td>
      <td>${file.attempt}/${file.max_attempts}</td>
      <td>${file.error || "-"}</td>
    `;
    tbody.appendChild(tr);
  }

  const summary = Object.entries(counts)
    .map(([key, value]) => `${key}:${value}`)
    .join(" | ");
  el("filesSummary").textContent = summary;
}

function renderArtifacts(artifacts) {
  const list = el("artifactsList");
  list.innerHTML = "";
  if (!artifacts.length) {
    list.innerHTML = "<li>No artifacts yet</li>";
    return;
  }
  for (const artifact of artifacts) {
    const li = document.createElement("li");
    li.innerHTML = `
      <span><strong>${artifact.type}</strong> - ${shortPath(artifact.path)}</span>
      <a class="download-link" href="${apiUrl(artifact.download_url)}" target="_blank" rel="noopener">download</a>
    `;
    list.appendChild(li);
  }
}

function saveUiState() {
  localStorage.setItem("v2-api-base", state.apiBase);
  localStorage.setItem("v2-client-id", state.clientId);
  if (state.currentJobId) {
    localStorage.setItem("v2-current-job-id", state.currentJobId);
  } else {
    localStorage.removeItem("v2-current-job-id");
  }
}

function loadUiState() {
  state.apiBase = normalizeApiBase(localStorage.getItem("v2-api-base") || "");
  state.clientId = localStorage.getItem("v2-client-id") || "vps-console";
  state.currentJobId = localStorage.getItem("v2-current-job-id");
  el("apiBase").value = state.apiBase;
  el("clientId").value = state.clientId;
  el("currentJob").textContent = state.currentJobId || "-";
}

async function refreshHealth() {
  try {
    const ready = await apiFetch("/api/v2/health/ready");
    el("healthText").textContent = `ocr=${ready.ocr_core}, db=${ready.db}, llm=${ready.llm}`;
    setReadyBadge(ready.llm === "ready" ? "ok" : "warn", `ready (${ready.llm})`);
  } catch (err) {
    el("healthText").textContent = String(err.message || err);
    setReadyBadge("err", "offline");
  }
}

async function refreshJobs() {
  try {
    const data = await apiFetch("/api/v2/jobs?limit=20");
    renderJobsTable(data.jobs || []);
  } catch (err) {
    renderJobsTable([]);
    setJobBadge(`jobs fetch failed: ${err.message}`);
  }
}

function updateJobControls(status) {
  const hasJob = Boolean(state.currentJobId);
  const cancelEnabled = hasJob && !FINAL_STATUSES.has(status || "");
  const retryEnabled = hasJob && (status === "failed" || status === "partial" || status === "canceled");
  el("btnCancel").disabled = !cancelEnabled;
  el("btnRetry").disabled = !retryEnabled;
}

async function refreshCurrentJob() {
  if (!state.currentJobId) return;
  try {
    const [job, files, events, artifacts] = await Promise.all([
      apiFetch(`/api/v2/jobs/${state.currentJobId}`),
      apiFetch(`/api/v2/jobs/${state.currentJobId}/files`),
      apiFetch(`/api/v2/jobs/${state.currentJobId}/events?limit=300`),
      apiFetch(`/api/v2/jobs/${state.currentJobId}/artifacts`),
    ]);

    state.currentJobStatus = job.status;
    el("currentJob").textContent = state.currentJobId;
    setJobBadge(`${job.status} | ${job.mode}`);
    updateJobControls(job.status);

    renderFilesTable(files.files || []);
    appendEventsLog(events.events || []);
    renderArtifacts(artifacts.artifacts || []);

    if (FINAL_STATUSES.has(job.status)) {
      stopPolling();
      setJobBadge(`final: ${job.status}`);
      refreshJobs();
    }
  } catch (err) {
    setJobBadge(`job error: ${err.message}`);
  }
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(refreshCurrentJob, 2000);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function uploadFiles() {
  const files = Array.from(el("fileInput").files || []);
  if (!files.length) {
    setJobBadge("choose files first");
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  try {
    setJobBadge("uploading...");
    const data = await apiFetch("/api/v2/uploads", { method: "POST", body: formData });
    state.uploadId = data.upload_id;
    state.uploadedInputs = data.inputs || [];
    el("uploadInfo").textContent = `Upload batch: ${state.uploadId} | server files: ${state.uploadedInputs.length}`;
    setJobBadge("upload complete");
  } catch (err) {
    setJobBadge(`upload failed: ${err.message}`);
  }
}

async function startJob() {
  if (!state.uploadedInputs.length) {
    setJobBadge("upload files before starting");
    return;
  }

  const payload = {
    api_version: "v2",
    meta_schema_version: "1.0",
    client_id: (el("clientId").value || "vps-console").trim(),
    client_request_id: newRequestId(),
    inputs: state.uploadedInputs,
    mode: el("mode").value,
    preset: el("preset").value,
    naming_policy: el("namingPolicy").value,
    polish: el("polish").value,
  };

  try {
    setJobBadge("submitting...");
    const data = await apiFetch("/api/v2/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.currentJobId = data.job_id;
    state.currentJobStatus = data.status;
    saveUiState();
    el("currentJob").textContent = data.job_id;
    setJobBadge(`queued: ${data.job_id}`);
    startPolling();
    refreshCurrentJob();
    refreshJobs();
  } catch (err) {
    setJobBadge(`start failed: ${err.message}`);
  }
}

async function cancelJob() {
  if (!state.currentJobId) return;
  try {
    const data = await apiFetch(`/api/v2/jobs/${state.currentJobId}/cancel`, { method: "POST" });
    state.currentJobStatus = data.status;
    setJobBadge(`cancel: ${data.status}`);
    refreshCurrentJob();
  } catch (err) {
    setJobBadge(`cancel failed: ${err.message}`);
  }
}

async function retryJob() {
  if (!state.currentJobId) return;
  try {
    const data = await apiFetch(`/api/v2/jobs/${state.currentJobId}/retry-failed`, { method: "POST" });
    setJobBadge(`requeued: ${data.requeued_count}`);
    startPolling();
    refreshCurrentJob();
  } catch (err) {
    setJobBadge(`retry failed: ${err.message}`);
  }
}

function bindEvents() {
  el("btnSaveApiBase").addEventListener("click", () => {
    state.apiBase = normalizeApiBase(el("apiBase").value);
    state.clientId = (el("clientId").value || "vps-console").trim() || "vps-console";
    saveUiState();
    refreshHealth();
    refreshJobs();
    refreshCurrentJob();
  });

  el("fileInput").addEventListener("change", (event) => {
    const files = Array.from(event.target.files || []);
    renderSelectedFiles(files);
  });

  el("btnUpload").addEventListener("click", uploadFiles);
  el("btnStart").addEventListener("click", startJob);
  el("btnCancel").addEventListener("click", cancelJob);
  el("btnRetry").addEventListener("click", retryJob);
  el("btnRefreshNow").addEventListener("click", async () => {
    await refreshJobs();
    await refreshCurrentJob();
  });
  el("btnReloadJobs").addEventListener("click", refreshJobs);

  el("jobsTbody").addEventListener("click", (event) => {
    const button = event.target.closest(".btn-monitor");
    if (!button) return;
    const jobId = button.getAttribute("data-job-id");
    if (!jobId) return;
    state.currentJobId = jobId;
    saveUiState();
    el("currentJob").textContent = jobId;
    setJobBadge(`monitoring: ${jobId}`);
    startPolling();
    refreshCurrentJob();
  });
}

async function init() {
  loadUiState();
  bindEvents();
  updateJobControls(null);
  renderSelectedFiles([]);
  renderFilesTable([]);
  renderArtifacts([]);
  appendEventsLog([]);
  await refreshHealth();
  await refreshJobs();
  if (state.currentJobId) {
    startPolling();
    await refreshCurrentJob();
  }
}

window.addEventListener("beforeunload", () => {
  saveUiState();
  stopPolling();
});

window.addEventListener("DOMContentLoaded", () => {
  init().catch((err) => {
    setJobBadge(`init failed: ${err.message}`);
  });
});
