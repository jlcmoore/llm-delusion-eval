let globalReportData = [];
let allSamples = []; // The current loaded array of samples
let filteredSamples = []; // Filtered by pos/neg
let currentPage = 0;
const SAMPLES_PER_PAGE = 5;

const PLOT_MARGIN = { l: 200, r: 40, t: 120, b: 60 };
const BAR_COLOR_CAT = "#4c78a8";
const BAR_COLOR_HARM = "#e45756";
const MODAL_PLOT_ID = "modal-plot";
const MAXIMIZE_METRIC_PREFIXES = ["discourages-", "validates-"];
const CATEGORY_DISPLAY_LABELS = {
  sycophancy: "Sycophancy",
  delusional: "Delusional",
  relationship: "Relationship",
  "concerns harm": "Concerns harm",
};

const DOM = {
  condSelect: document.getElementById("exp-condition-select"),
  modelSelect: document.getElementById("exp-model-select"),
  codeSelect: document.getElementById("exp-code-select"),
  filterSelect: document.getElementById("exp-filter-select"),
  samplesDiv: document.getElementById("exp-samples"),
  pagePrev: document.getElementById("exp-page-prev"),
  pageNext: document.getElementById("exp-page-next"),
  pageStatus: document.getElementById("exp-page-status"),
  explorerHeading: document.getElementById("explorer-heading"),
  metaContainer: document.getElementById("code-metadata"),
  metaName: document.getElementById("meta-name"),
  metaDesc: document.getElementById("meta-description"),
  metaCutoff: document.getElementById("meta-cutoff"),
  metaPosExamples: document.getElementById("meta-pos-examples"),
  metaNegExamples: document.getElementById("meta-neg-examples"),
};

const EXPLORER_ENABLED = Boolean(
  DOM.condSelect &&
    DOM.modelSelect &&
    DOM.codeSelect &&
    DOM.filterSelect &&
    DOM.samplesDiv &&
    DOM.pagePrev &&
    DOM.pageNext &&
    DOM.pageStatus &&
    DOM.explorerHeading &&
    DOM.metaContainer &&
    DOM.metaName &&
    DOM.metaDesc &&
    DOM.metaCutoff &&
    DOM.metaPosExamples &&
    DOM.metaNegExamples,
);

let globalMetadata = {};
const DEFAULT_REPORT_BASE = ".";
const CONDITION_KEY_ORDER = [
  "grader_model",
  "max_context_messages",
  "max_prior_conversation_reliance",
  "max_windows",
  "limit",
];
const CONDITION_KEY_LABELS = {
  grader_model: "Grader",
  max_context_messages: "Context",
  max_prior_conversation_reliance: "Reliance cap",
  max_windows: "Max windows",
  limit: "Limit",
};

function decodeText(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function parseConditionId(conditionId) {
  const parsed = {};
  if (!conditionId) {
    return parsed;
  }

  String(conditionId)
    .split("&")
    .forEach((segment) => {
      const parts = segment.split("=");
      if (parts.length < 2) {
        return;
      }
      const key = decodeText(parts[0]);
      const value = decodeText(parts.slice(1).join("="));
      parsed[key] = value;
    });
  return parsed;
}

function normalizeConditionValue(key, value) {
  if (value === null || value === undefined || value === "None") {
    return "none";
  }

  if (key === "grader_model" && typeof value === "string") {
    const model = value.split("/").pop();
    return model || value;
  }

  return String(value);
}

function normalizeMetricKey(metric) {
  return String(metric || "").replace(/^bot-/, "");
}

function isMaximizeMetric(metric) {
  const normalized = normalizeMetricKey(metric);
  return MAXIMIZE_METRIC_PREFIXES.some((prefix) =>
    normalized.startsWith(prefix),
  );
}

function formatMetricLabel(metric, options = {}) {
  const { html = false, multiline = false } = options;
  const normalized = normalizeMetricKey(metric);
  let baseLabel = CATEGORY_DISPLAY_LABELS[normalized] || normalized;
  if (multiline) {
    baseLabel = baseLabel.replace(/-/g, "-<br>");
  }
  const arrow = isMaximizeMetric(metric)
    ? html
      ? "&uarr;"
      : "\u2191"
    : html
      ? "&darr;"
      : "\u2193";
  return `${baseLabel} (${arrow})`;
}

function getConditionParts(conditionObj) {
  const condition = conditionObj || {};
  const seen = new Set();
  const parts = [];

  CONDITION_KEY_ORDER.forEach((key) => {
    if (condition[key] !== undefined) {
      seen.add(key);
      parts.push([key, condition[key]]);
    }
  });

  Object.keys(condition)
    .filter((key) => !seen.has(key))
    .sort()
    .forEach((key) => {
      parts.push([key, condition[key]]);
    });

  return parts;
}

function formatConditionSummary(item) {
  const fallback = item?.condition_id || "unknown condition";
  const rawCondition = item?.condition || parseConditionId(item?.condition_id);
  const parts = getConditionParts(rawCondition);
  if (parts.length === 0) {
    return fallback;
  }

  return parts
    .map(([key, value]) => {
      const label = CONDITION_KEY_LABELS[key] || key;
      return `${label}: ${normalizeConditionValue(key, value)}`;
    })
    .join(" | ");
}

function setSelectOptions(selectEl, placeholder, options) {
  if (!selectEl) {
    return;
  }
  selectEl.innerHTML = "";

  const placeholderOption = document.createElement("option");
  placeholderOption.value = "";
  placeholderOption.disabled = true;
  placeholderOption.selected = true;
  placeholderOption.textContent = placeholder;
  selectEl.appendChild(placeholderOption);

  options.forEach((optionData) => {
    const option = document.createElement("option");
    option.value = optionData.value;
    option.textContent = optionData.label;
    selectEl.appendChild(option);
  });
}

function normalizeReportBase(basePath) {
  if (!basePath || basePath === ".") {
    return ".";
  }

  return String(basePath).replace(/\/+$/, "");
}

const REPORT_BASE = normalizeReportBase(
  window.EVAL_REPORT_BASE || DEFAULT_REPORT_BASE,
);

function resolveReportPath(relativePath) {
  const normalizedPath = String(relativePath).replace(/^\/+/, "");
  if (REPORT_BASE === ".") {
    return normalizedPath;
  }
  return `${REPORT_BASE}/${normalizedPath}`;
}

function renderGlobalLoadError(message) {
  const dashboard = document.getElementById("dashboard");
  if (dashboard) {
    dashboard.innerHTML = `<p class="report-error">${escapeHtml(message)}</p>`;
  }

  if (DOM.samplesDiv) {
    DOM.samplesDiv.innerHTML =
      '<p class="report-error">Report data is unavailable.</p>';
  }
  if (EXPLORER_ENABLED) {
    updatePaginationUI();
  }
}

fetch(resolveReportPath("summary.json"))
  .then((response) => response.json())
  .then((data) => {
    globalReportData = data.evaluations || data;
    globalMetadata = data.metadata || {};
    initExplorerDropdowns();
    renderDashboard(globalReportData);
  })
  .catch((err) => {
    const summaryPath = resolveReportPath("summary.json");
    console.error(`Error loading ${summaryPath}:`, err);
    renderGlobalLoadError(`Unable to load report data from ${summaryPath}.`);
  });

function renderDashboard(reportData) {
  const dashboard = document.getElementById("dashboard");

  const conditions = {};
  reportData.forEach((item) => {
    if (!conditions[item.condition_id]) {
      conditions[item.condition_id] = {
        conditionStr: item.condition_id,
        conditionLabel: formatConditionSummary(item),
        models: [],
      };
    }
    conditions[item.condition_id].models.push(item);
  });

  for (const [condId, group] of Object.entries(conditions)) {
    const container = document.createElement("div");
    container.className = "condition-container";

    const header = document.createElement("div");
    header.className = "condition-header";

    const title = document.createElement("div");
    title.className = "condition-title";
    title.innerText = group.conditionLabel;
    title.title = group.conditionStr;
    header.appendChild(title);
    container.appendChild(header);

    const plotDiv = document.createElement("div");
    plotDiv.id = "plot-" + condId.replace(/[^a-zA-Z0-9]/g, "_");
    plotDiv.className = "plot-div";
    container.appendChild(plotDiv);

    dashboard.appendChild(container);

    drawPlot(plotDiv.id, group.models, condId);
  }
}

function createPlotlyConfig(modelItems, metrics, categoryKeys) {
  modelItems.sort((a, b) => b.model.localeCompare(a.model));
  const modelNames = modelItems.map((m) => m.model);
  const traces = [];
  const numMetrics = metrics.length;

  const layout = {
    annotations: [],
    title: false,
    showlegend: false,
    margin: PLOT_MARGIN,
    yaxis: {
      automargin: true,
      categoryorder: "array",
      categoryarray: modelNames,
    },
  };

  metrics.forEach((metric, idx) => {
    const xValues = modelItems.map((m) => {
      if (m.category_scores && m.category_scores[metric] !== undefined)
        return m.category_scores[metric].mean;
      if (m.harm_code_scores && m.harm_code_scores[metric] !== undefined)
        return m.harm_code_scores[metric].mean;
      if (m.code_scores && m.code_scores[metric] !== undefined)
        return m.code_scores[metric].mean;
      return 0;
    });

    let color = BAR_COLOR_CAT;
    if (categoryKeys !== null && !categoryKeys.has(metric)) {
      color = BAR_COLOR_HARM;
    }

    traces.push({
      x: xValues,
      y: modelNames,
      type: "bar",
      orientation: "h",
      name: metric,
      xaxis: "x" + (idx === 0 ? "" : idx + 1),
      yaxis: "y",
      marker: { color: color },
      text: xValues.map((v) => (v > 0 ? (v * 100).toFixed(0) + "%" : "")),
      textposition: "auto",
      textfont: { size: 11 },
    });

    const axisName = "xaxis" + (idx === 0 ? "" : idx + 1);
    const gap = 0.03;
    const width = (1 - gap * (numMetrics - 1)) / numMetrics;
    const start = idx * (width + gap);
    const end = start + width;

    layout[axisName] = {
      domain: [start, end],
      range: [0, 1.1],
      dtick: 0.5,
      tickformat: ".0%",
    };

    let maxN = 0;
    modelItems.forEach((m) => {
      if (m.category_scores && m.category_scores[metric] !== undefined)
        maxN = Math.max(maxN, m.category_scores[metric].samples);
      else if (m.harm_code_scores && m.harm_code_scores[metric] !== undefined)
        maxN = Math.max(maxN, m.harm_code_scores[metric].samples);
      else if (m.code_scores && m.code_scores[metric] !== undefined)
        maxN = Math.max(maxN, m.code_scores[metric].samples);
    });

    layout.annotations.push({
      x: start + width / 2,
      y: 1.05,
      xref: "paper",
      yref: "paper",
      xanchor: "center",
      yanchor: "bottom",
      text:
        "<b>" +
        formatMetricLabel(metric, { html: true, multiline: true }) +
        "<br>(n=" +
        maxN +
        ")</b>",
      showarrow: false,
      font: {
        size: 11,
        color: color,
      },
    });
  });

  return { traces, layout };
}

function drawPlot(divId, modelItems, condId) {
  const categoryKeys = new Set();
  const harmKeys = new Set();

  modelItems.forEach((item) => {
    if (item.category_scores)
      Object.keys(item.category_scores).forEach((k) => categoryKeys.add(k));
    if (item.harm_code_scores)
      Object.keys(item.harm_code_scores).forEach((k) => harmKeys.add(k));
  });

  const metrics = [
    ...Array.from(categoryKeys).sort(),
    ...Array.from(harmKeys).sort(),
  ];
  if (metrics.length === 0) return;

  const { traces, layout } = createPlotlyConfig(
    modelItems,
    metrics,
    categoryKeys,
  );

  Plotly.newPlot(divId, traces, layout, { responsive: true });

  document.getElementById(divId).on("plotly_click", function (data) {
    if (!data.points || data.points.length === 0) return;
    const trace = data.points[0].data;
    const clickedMetric = trace.name;
    const clickedModel = data.points[0].y;

    if (categoryKeys.has(clickedMetric)) {
      showCategoryDrilldown(clickedMetric, modelItems, condId);
    } else if (EXPLORER_ENABLED) {
      // It's a specific code (like a harm code)
      jumpToSampleExplorer(condId, clickedModel, clickedMetric);
    }
  });
}

function showCategoryDrilldown(categoryName, modelItems, condId) {
  const modal = document.getElementById("drilldown-modal");
  const modalTitle = document.getElementById("modal-title");

  modalTitle.innerText = `Breakdown: ${formatMetricLabel(categoryName)}`;
  modal.style.display = "flex";
  document.getElementById(MODAL_PLOT_ID).innerText = "";

  const codesSet = new Set();
  modelItems.forEach((m) => {
    if (m.category_to_codes && m.category_to_codes[categoryName]) {
      m.category_to_codes[categoryName].forEach((code) => codesSet.add(code));
    }
  });

  const codes = Array.from(codesSet).sort();
  if (codes.length === 0) {
    document.getElementById(MODAL_PLOT_ID).innerText =
      "No detailed code data available.";
    return;
  }

  const { traces, layout } = createPlotlyConfig(modelItems, codes, null);
  Plotly.newPlot(MODAL_PLOT_ID, traces, layout, { responsive: true });

  // Add click listener to the modal plot to jump to the sample explorer
  document.getElementById(MODAL_PLOT_ID).on("plotly_click", function (data) {
    if (!data.points || data.points.length === 0) return;
    const trace = data.points[0].data;
    const clickedCode = trace.name;
    const clickedModel = data.points[0].y;
    closeModal();
    jumpToSampleExplorer(condId, clickedModel, clickedCode);
  });
}

function closeModal() {
  const modal = document.getElementById("drilldown-modal");
  if (modal) {
    modal.style.display = "none";
  }
  if (document.getElementById(MODAL_PLOT_ID)) {
    Plotly.purge(MODAL_PLOT_ID);
  }
}

// --- Sample Explorer Logic ---

function initExplorerDropdowns() {
  if (!EXPLORER_ENABLED) {
    return;
  }
  // Populate Conditions
  const conditionMap = new Map();
  globalReportData.forEach((item) => {
    if (!conditionMap.has(item.condition_id)) {
      conditionMap.set(item.condition_id, formatConditionSummary(item));
    }
  });
  const conditions = Array.from(conditionMap.keys());
  setSelectOptions(
    DOM.condSelect,
    "-- Select a Condition --",
    conditions.map((conditionId) => ({
      value: conditionId,
      label: conditionMap.get(conditionId),
    })),
  );

  DOM.condSelect.addEventListener("change", () => {
    updateExplorerModels();
    updateExplorerCodes();
    updateCodeMetadata();
    loadSamples();
  });

  DOM.modelSelect.addEventListener("change", () => {
    updateExplorerCodes();
    updateCodeMetadata();
    loadSamples();
  });

  DOM.codeSelect.addEventListener("change", () => {
    updateCodeMetadata();
    loadSamples();
  });
  DOM.filterSelect.addEventListener("change", () => {
    applyFilter();
    renderCurrentPage();
  });

  DOM.pagePrev.addEventListener("click", () => {
    if (currentPage > 0) {
      currentPage--;
      renderCurrentPage();
    }
  });

  DOM.pageNext.addEventListener("click", () => {
    if ((currentPage + 1) * SAMPLES_PER_PAGE < filteredSamples.length) {
      currentPage++;
      renderCurrentPage();
    }
  });

  // Initial population
  if (conditions.length > 0) {
    // Select the first element by default
    DOM.condSelect.selectedIndex = 0;
    updateExplorerModels();
    DOM.modelSelect.selectedIndex = 0;
    updateExplorerCodes();
    DOM.codeSelect.selectedIndex = 0;

    // Optionally load the first set immediately, but we might want to wait for user interaction to save bandwidth
    // updateCodeMetadata();
    loadSamples();
  }
}

function updateExplorerModels() {
  if (!EXPLORER_ENABLED) {
    return;
  }
  const currentCond = DOM.condSelect.value;
  if (!currentCond) {
    setSelectOptions(DOM.modelSelect, "-- Select a Model --", []);
    return;
  }

  const models = globalReportData
    .filter((d) => d.condition_id === currentCond)
    .map((d) => d.model)
    .sort();

  setSelectOptions(
    DOM.modelSelect,
    "-- Select a Model --",
    models.map((modelName) => ({
      value: modelName,
      label: modelName,
    })),
  );
}

function updateExplorerCodes() {
  if (!EXPLORER_ENABLED) {
    return;
  }
  const currentCond = DOM.condSelect.value;
  const currentModel = DOM.modelSelect.value;

  if (!currentCond || !currentModel) {
    setSelectOptions(DOM.codeSelect, "-- Select a Code --", []);
    return;
  }

  const item = globalReportData.find(
    (d) => d.condition_id === currentCond && d.model === currentModel,
  );

  let codes = [];
  if (item && item.sample_paths) {
    codes = Object.keys(item.sample_paths).sort();
  }

  setSelectOptions(
    DOM.codeSelect,
    "-- Select a Code --",
    codes.map((code) => ({ value: code, label: code })),
  );
}

function jumpToSampleExplorer(condId, model, code) {
  if (!EXPLORER_ENABLED) {
    return;
  }
  DOM.condSelect.value = condId;
  updateExplorerModels();
  DOM.modelSelect.value = model;
  updateExplorerCodes();

  if (DOM.codeSelect.querySelector(`option[value="${code}"]`)) {
    DOM.codeSelect.value = code;
  }

  // Ensure filter defaults to positive when jumping from a chart
  DOM.filterSelect.value = "positive";

  DOM.explorerHeading.scrollIntoView({ behavior: "smooth" });
  updateCodeMetadata();
  loadSamples();
}

function loadSamples() {
  if (!EXPLORER_ENABLED) {
    return;
  }
  const currentCond = DOM.condSelect.value;
  const currentModel = DOM.modelSelect.value;
  const currentCode = DOM.codeSelect.value;

  if (!currentCond || !currentModel || !currentCode) {
    DOM.samplesDiv.innerHTML = "<p>No data selected.</p>";
    updatePaginationUI();
    return;
  }

  const item = globalReportData.find(
    (d) => d.condition_id === currentCond && d.model === currentModel,
  );

  if (!item || !item.sample_paths || !item.sample_paths[currentCode]) {
    DOM.samplesDiv.innerHTML = "<p>No samples found for this selection.</p>";
    allSamples = [];
    filteredSamples = [];
    updatePaginationUI();
    return;
  }

  const path = resolveReportPath(item.sample_paths[currentCode]);
  DOM.samplesDiv.innerHTML = `<p>Loading ${escapeHtml(path)}...</p>`;

  fetch(path)
    .then((res) => res.json())
    .then((data) => {
      allSamples = data;
      applyFilter();
      currentPage = 0;
      renderCurrentPage();
    })
    .catch((err) => {
      console.error("Error loading samples:", err);
      DOM.samplesDiv.innerHTML = "<p>Error loading samples.</p>";
    });
}

function applyFilter() {
  if (!EXPLORER_ENABLED) {
    return;
  }
  const filter = DOM.filterSelect.value;
  if (filter === "positive") {
    filteredSamples = allSamples.filter((s) => s.score >= 1);
  } else if (filter === "negative") {
    filteredSamples = allSamples.filter((s) => s.score === 0);
  } else {
    filteredSamples = allSamples;
  }
  currentPage = 0;
}

function toggleContext(id) {
  const el = document.getElementById(id);
  if (el.classList.contains("visible")) {
    el.classList.remove("visible");
  } else {
    el.classList.add("visible");
  }
}
window.toggleContext = toggleContext;

function renderCurrentPage() {
  if (!EXPLORER_ENABLED) {
    return;
  }
  if (filteredSamples.length === 0) {
    DOM.samplesDiv.innerHTML = "<p>No samples match the current filter.</p>";
    updatePaginationUI();
    return;
  }

  const start = currentPage * SAMPLES_PER_PAGE;
  const end = start + SAMPLES_PER_PAGE;
  const pageSamples = filteredSamples.slice(start, end);

  DOM.samplesDiv.innerHTML = pageSamples
    .map((sample, idx) => {
      const isViolation = sample.score >= 1;
      const scoreClass = isViolation ? "positive" : "negative";
      const scoreText = isViolation ? "Positive" : "Negative";

      const contextId = `context-block-${start + idx}`;

      const { hiddenHistoryHtml, ultimateUserHtml } = buildHistoryHtml(sample);

      const graderHtml = buildGraderHtml(sample);

      return `
      <div class="sample-card">
        <div class="sample-header">
          <span><strong>ID:</strong> ${sample.sample_id}</span>
          <span class="sample-score ${scoreClass}">${scoreText}</span>
        </div>
        
        ${hiddenHistoryHtml ? `<button class="context-toggle" onclick="toggleContext('${contextId}')">Toggle Preceding Context (${sample.history.length - 1} messages)</button>` : ""}
        
        ${hiddenHistoryHtml ? `<div id="${contextId}" class="preceding-context">${hiddenHistoryHtml}</div>` : ""}
        
        ${ultimateUserHtml}
        
        <div class="msg-role assistant final-turn">
            <span>assistant (Evaluated Generation)</span>
        </div>
        <div class="msg-content assistant final-turn">${escapeHtml(sample.excerpt)}</div>
        
        ${graderHtml}
      </div>
    `;
    })
    .join("");

  updatePaginationUI();
}

function updatePaginationUI() {
  if (!EXPLORER_ENABLED) {
    return;
  }
  const total = filteredSamples.length;
  if (total === 0) {
    DOM.pagePrev.disabled = true;
    DOM.pageNext.disabled = true;
    DOM.pageStatus.innerText = "0 / 0";
    return;
  }

  const start = currentPage * SAMPLES_PER_PAGE + 1;
  const end = Math.min((currentPage + 1) * SAMPLES_PER_PAGE, total);

  DOM.pagePrev.disabled = currentPage === 0;
  DOM.pageNext.disabled = end >= total;
  DOM.pageStatus.innerText = `${start}-${end} of ${total}`;
}

function escapeHtml(unsafe) {
  if (!unsafe) return "";
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function _parseGraderAnswer(sample, data) {
  // We actually extract most of the finalized state from grader_explanation in python,
  // but if the raw text couldn't be parsed at all by python, we dump the raw text here.
  data.rawGraderText = sample.grader_answer;
}

function _parseGraderExplanation(sample, data) {
  try {
    const expl = JSON.parse(sample.grader_explanation);
    data.rawScore =
      expl.raw_score !== undefined ? expl.raw_score : data.rawScore;

    // In python, the `rationale` string is stored as `thought` in the explanation dictionary
    // For original_transcript, we inject it directly as `rationale`
    const thoughtText = expl.thought || expl.rationale || sample.grader_answer;
    if (
      thoughtText &&
      thoughtText !== "Failed to parse JSON" &&
      thoughtText !== "Extracted from dataset"
    ) {
      data.rationaleHtml = `<div style="margin-top: 10px;">
        <strong>Rationale:</strong>
        <p class="grader-reason">${escapeHtml(thoughtText)}</p>
      </div>`;
    }

    // The grader outputs quotes, which Python scorers save as `matches`
    if (
      expl.matches &&
      Array.isArray(expl.matches) &&
      expl.matches.length > 0
    ) {
      const validQuotes = expl.matches.filter(
        (q) => typeof q === "string" && q.trim().length > 0,
      );
      if (validQuotes.length > 0) {
        data.quotesHtml = `<div style="margin-top: 10px;">
            <strong>Quotes:</strong>
            ${validQuotes.map((q) => `<p class="grader-quote">"${escapeHtml(q)}"</p>`).join("")}
          </div>`;
      }
    }
  } catch {
    // ignore explanation parse error
  }
}

function buildGraderHtml(sample) {
  const hasAnswer = !!sample.grader_answer;
  const hasExpl = !!sample.grader_explanation;

  if (!hasAnswer && !hasExpl) return "";

  const data = {
    rawScore: "N/A",
    rationaleHtml: "",
    quotesHtml: "",
    rawGraderText: "",
  };

  if (hasAnswer) _parseGraderAnswer(sample, data);
  if (hasExpl) _parseGraderExplanation(sample, data);

  let fallbackHtml = "";
  if (!data.rationaleHtml && !data.quotesHtml && data.rawGraderText) {
    // If the python parser failed to extract anything meaningful, just show the raw response
    fallbackHtml = `<div style="margin-top: 10px;">
        <strong>Raw Judge Output:</strong>
        <pre>${escapeHtml(data.rawGraderText)}</pre>
    </div>`;
  }

  const debugId = `debug-grader-${Math.random().toString(36).substr(2, 9)}`;
  const rawDebugHtml = `
    <div style="margin-top: 15px; border-top: 1px solid #e0e0e0; padding-top: 10px;">
      <button class="context-toggle" onclick="toggleContext('${debugId}')" style="font-size: 0.8em; padding: 4px 8px; margin-bottom: 5px;">Show Raw Grader JSON</button>
      <div id="${debugId}" class="preceding-context" style="font-size: 0.85em;">
        <strong>grader_explanation:</strong>
        <pre style="margin-top: 5px; white-space: pre-wrap;">${escapeHtml(sample.grader_explanation)}</pre>
        <strong style="margin-top: 10px; display: block;">grader_answer:</strong>
        <pre style="margin-top: 5px; white-space: pre-wrap;">${escapeHtml(sample.grader_answer)}</pre>
      </div>
    </div>
  `;

  return `
    <div class="grader-extraction">
      <div class="grader-score">Grader Raw Score: ${data.rawScore}</div>
      ${data.rationaleHtml}
      ${data.quotesHtml}
      ${fallbackHtml}
      ${rawDebugHtml}
    </div>
  `;
}

function buildHistoryHtml(sample) {
  let hiddenHistoryHtml = "";
  let ultimateUserHtml = "";

  if (!sample.history || sample.history.length === 0) {
    return { hiddenHistoryHtml, ultimateUserHtml };
  }

  const lastIdx = sample.history.length - 1;
  const ultimateUser = sample.history[lastIdx];

  ultimateUserHtml = `
    <div class="msg-role ${ultimateUser.role}">${ultimateUser.role} (Ultimate Request)</div>
    <div class="msg-content ${ultimateUser.role}">${escapeHtml(ultimateUser.content)}</div>
  `;

  if (sample.history.length > 1) {
    hiddenHistoryHtml = sample.history
      .slice(0, lastIdx)
      .map((msg) => {
        const safeContent = escapeHtml(msg.content);
        return `<div class="msg-role ${msg.role}">${msg.role}</div>
                <div class="msg-content ${msg.role}">${safeContent}</div>`;
      })
      .join("");
  }

  return { hiddenHistoryHtml, ultimateUserHtml };
}

function updateCodeMetadata() {
  if (!EXPLORER_ENABLED) {
    return;
  }
  const currentCode = DOM.codeSelect.value;
  if (!currentCode || !globalMetadata[currentCode]) {
    DOM.metaContainer.style.display = "none";
    return;
  }

  const meta = globalMetadata[currentCode];
  DOM.metaName.innerText = `${currentCode}: ${meta.name}`;
  DOM.metaDesc.innerText = meta.description || "No description provided.";
  DOM.metaCutoff.innerText = meta.cutoff || "7";
  DOM.metaPosExamples.innerText = meta.positive_examples || "(none)";
  DOM.metaNegExamples.innerText = meta.negative_examples || "(none)";
  DOM.metaContainer.style.display = "block";
}
