const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("outputTable") || document.getElementById("outputText");
const fileInfoEl = document.getElementById("fileInfo");
const previewImage = document.getElementById("previewImage");

if (!fileInput || !analyzeBtn || !statusEl || !outputEl || !fileInfoEl || !previewImage) {
  console.error("UI elements not found");
}

let previewUrl = null;

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

function setFileInfo(text) {
  if (fileInfoEl) fileInfoEl.textContent = text;
}

function setOutputMessage(text) {
  if (!outputEl) return;
  if (outputEl.tagName === "PRE") {
    outputEl.textContent = text;
    return;
  }
  outputEl.innerHTML = `<div class="output-empty">${text}</div>`;
}

function renderTable(rows) {
  if (!outputEl) return;
  if (outputEl.tagName === "PRE") {
    outputEl.textContent = rows && rows.length ? rows.map((row) => `${row.num}. ${row.text}`).join("\n") : "Пустой ответ.";
    return;
  }
  outputEl.innerHTML = "";
  if (!rows || rows.length === 0) {
    setOutputMessage("Пустой ответ.");
    return;
  }

  const hasActor = rows.some((row) => row.actor);

  const table = document.createElement("table");
  table.className = "result-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");

  const thNum = document.createElement("th");
  thNum.textContent = "№";
  thNum.className = "num-cell";
  headRow.appendChild(thNum);

  const thText = document.createElement("th");
  thText.textContent = hasActor ? "Наименование действия" : "Шаг";
  headRow.appendChild(thText);

  if (hasActor) {
    const thActor = document.createElement("th");
    thActor.textContent = "Роль";
    headRow.appendChild(thActor);
  }

  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.kind === "condition") tr.classList.add("condition-row");
    if (row.kind === "branch") tr.classList.add("branch-row");

    const tdNum = document.createElement("td");
    tdNum.className = "num-cell";
    tdNum.textContent = row.num || "";
    tr.appendChild(tdNum);

    const tdText = document.createElement("td");
    const depth = row.num ? row.num.split(".").length - 1 : 0;
    tdText.style.paddingLeft = `${12 + depth * 12}px`;
    tdText.textContent = row.text || "";
    tr.appendChild(tdText);

    if (hasActor) {
      const tdActor = document.createElement("td");
      tdActor.textContent = row.actor ? row.actor : "—";
      tr.appendChild(tdActor);
    }

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  outputEl.appendChild(table);
}

fileInput?.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    setFileInfo("Файл не выбран");
    previewImage.style.display = "none";
    previewImage.removeAttribute("src");
    return;
  }

  setFileInfo(`Выбран файл: ${file.name}`);
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.style.display = "block";
  setStatus("Файл готов к анализу");
});

analyzeBtn?.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Сначала выберите PNG/JPG/SVG файл.");
    return;
  }

  setOutputMessage("Идёт обработка...");
  setStatus("Загрузка...");
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  const start = performance.now();
  const controller = new AbortController();
  const timeoutMs = 30000;
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : payload;
      throw new Error(detail || "Ошибка запроса");
    }

    renderTable(payload.rows || []);
    const elapsed = ((performance.now() - start) / 1000).toFixed(2);
    setStatus(`Готово за ${elapsed}с`);
  } catch (err) {
    if (err.name === "AbortError") {
      setOutputMessage("Превышено время ожидания. Попробуйте уменьшить изображение или проверьте GPU.");
      setStatus("Таймаут");
    } else {
      setOutputMessage("Ошибка: " + err.message);
      setStatus("Ошибка");
    }
  } finally {
    clearTimeout(timeoutId);
    analyzeBtn.disabled = false;
  }
});

setOutputMessage("Выберите файл и нажмите «Распознать»." );
