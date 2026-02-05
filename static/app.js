const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("outputText");
const fileInfoEl = document.getElementById("fileInfo");
const previewImage = document.getElementById("previewImage");

let previewUrl = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function setFileInfo(text) {
  fileInfoEl.textContent = text;
}

fileInput.addEventListener("change", () => {
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

analyzeBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Сначала выберите PNG или JPG файл.");
    return;
  }

  outputEl.textContent = "Идёт обработка...";
  setStatus("Загрузка...");
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  const start = performance.now();
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    const text = await response.text();
    if (!response.ok) {
      throw new Error(text);
    }

    outputEl.textContent = text || "Пустой ответ.";
    const elapsed = ((performance.now() - start) / 1000).toFixed(2);
    setStatus(`Готово за ${elapsed}с`);
  } catch (err) {
    outputEl.textContent = "Ошибка: " + err.message;
    setStatus("Ошибка");
  } finally {
    analyzeBtn.disabled = false;
  }
});
