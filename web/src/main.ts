import katex from "katex";
import "katex/dist/katex.min.css";
import "./style.css";

type Prediction = {
  latex: string;
  tokens: string[];
  latency_ms: number;
  image_size: number[];
  source_size: number[];
  input_image: string;
};

const input = document.querySelector<HTMLInputElement>("#image-input")!;
const status = document.querySelector<HTMLParagraphElement>("#status")!;
const result = document.querySelector<HTMLElement>("#result")!;
const latex = document.querySelector<HTMLTextAreaElement>("#latex")!;
const formula = document.querySelector<HTMLDivElement>("#formula")!;
let activeRequest = 0;

async function recognize(file: File): Promise<void> {
  const requestId = ++activeRequest;
  status.textContent = "Đang nhận dạng…";
  result.hidden = true;
  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch("/api/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail ?? "Không thể nhận dạng ảnh");
    if (requestId !== activeRequest) return;
    const prediction = data as Prediction;
    latex.value = prediction.latex;
    renderFormula(prediction.latex);
    document.querySelector<HTMLSpanElement>("#latency")!.textContent = `${prediction.latency_ms} ms`;
    document.querySelector<HTMLImageElement>("#processed")!.src = prediction.input_image;
    document.querySelector<HTMLParagraphElement>("#shape")!.textContent =
      `Ảnh gốc ${prediction.source_size.join(" × ")} px → canvas model nhận ${prediction.image_size.join(" × ")} px`;
    document.querySelector<HTMLElement>("#tokens")!.textContent = prediction.tokens.join("  ");
    result.hidden = false;
    status.textContent = "";
  } catch (error) {
    if (requestId === activeRequest) {
      status.textContent = error instanceof Error ? error.message : "Đã có lỗi xảy ra";
    }
  }
}

function renderFormula(value: string): void {
  try {
    katex.render(value, formula, { displayMode: true, throwOnError: false });
  } catch {
    formula.textContent = value;
  }
}

input.addEventListener("change", () => {
  const file = input.files?.[0];
  result.hidden = true;
  status.textContent = "";
  if (file) {
    void recognize(file);
  }
});

const dropZone = document.querySelector<HTMLLabelElement>(".drop-zone")!;
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const files = event.dataTransfer?.files;
  const file = files?.[0];
  if (!file) return;
  input.files = files;
  input.dispatchEvent(new Event("change"));
});

latex.addEventListener("input", () => renderFormula(latex.value));
document.querySelector<HTMLButtonElement>("#copy")!.addEventListener("click", async () => {
  await navigator.clipboard.writeText(latex.value);
  status.textContent = "Đã sao chép LaTeX.";
});
