# ĐỒ ÁN CS232 – NHẬN DẠNG CÔNG THỨC TOÁN THÀNH LaTeX

Tài liệu này mô tả **hệ thống đang chạy thật trong repo**, không phải kế hoạch dự kiến. Mọi con số tensor, token và độ trễ trong đây được lấy bằng cách chạy pipeline thật trên một ảnh thật (xem mục 2) bằng `.venv\Scripts\python.exe`, không có số liệu bịa.

## 0. Thay đổi so với đề cương ban đầu

Đề cương ban đầu yêu cầu tự huấn luyện CNN encoder + Transformer decoder trên CROHME và cấm dùng mô hình pretrained làm giải pháp chính. Hướng đó **đã bị thay** bằng mô hình pretrained [UniMERNet](https://github.com/opendatalab/UniMERNet) (`wanderkid/unimernet_tiny`), vì:

- Không có GPU và không có CROHME 2023 đã tải/giải nén để huấn luyện thật; baseline tự train chưa từng đạt checkpoint có ý nghĩa.
- Phần cần bảo vệ (giải thích một ảnh biến thành từng token LaTeX thế nào) vẫn giữ nguyên được, chỉ là giải thích trên mô hình pretrained.

Hệ quả: toàn bộ code train/dataset/tokenizer/preprocess tự viết đã bị xóa khỏi `src/` (còn lưu trong `notebooks/` và `docs/` dưới dạng tài liệu nghiên cứu). Mục 9 liệt kê chi tiết những gì đã bỏ.

Hệ thống hiện tại = **upload ảnh → FastAPI → UniMERNet tiny → LaTeX → KaTeX render trong browser**.

## 1. Mục tiêu hiện tại

Web app nhận ảnh công thức toán (in hoặc viết tay), trả về mã LaTeX, render lại bằng KaTeX, cho phép sửa và copy mã LaTeX, kèm thông tin debug (độ trễ, kích thước ảnh, chuỗi token).

Không mục tiêu: huấn luyện, đánh giá ExpRate trên tập test, ablation positional encoding, so sánh độ phân giải. Những phần đó thuộc kế hoạch cũ và không còn code hỗ trợ.

## 2. Data point xuyên suốt tài liệu

Mọi mục dưới đây dùng **cùng một mẫu**:

- **Nhãn LaTeX:** `\frac{x^2}{y+1}`
- **Ảnh:** PNG 300×160 RGB, nền trắng, công thức đen ở giữa (render bằng mathtext để có một ảnh thật, thay cho mẫu CROHME vì repo không chứa dataset).

Gọi ảnh này là **mẫu A**. Bảng ở mục 4 là đường đi thật của mẫu A qua từng tầng, kèm shape thật in ra từ model.

## 3. Kiến trúc hệ thống

```text
Browser (web/src/main.ts)
   │  FormData: file ảnh
   ▼
POST /api/predict  (src/api.py, FastAPI)
   │  validate content-type / 10 MB / 25 M pixel
   ▼
FormulaImageEvalProcessor  (unimernet.processors, image_size=[192, 672])
   │  crop margin → resize giữ tỉ lệ → pad giữa → grayscale → normalize → tensor
   ▼
UniMERModel "unimernet_tiny"  (107.4 M tham số, tải 1 lần qua lru_cache)
   │  Donut-Swin encoder  →  mBart-style decoder (greedy, autoregressive)
   ▼
pred_str  "\frac { x ^ { 2 } } { y + 1 }"
   │  TOKEN_PATTERN tách token để debug
   ▼
JSON {latex, tokens, latency_ms, image_size, input_image}
   │
   ▼
KaTeX render + textarea cho phép sửa (web/src/main.ts)
```

Mô hình chỉ được nạp ở lần gọi `/api/predict` đầu tiên (`@lru_cache(maxsize=1)` trong [src/api.py](src/api.py)), nên khởi động server vẫn nhanh; checkpoint `unimernet_tiny.pth` (~430 MB) nằm trong Hugging Face cache, không nằm trong repo.

## 4. Mẫu A đi qua từng tầng

| # | Tầng | Mẫu A trước | Mẫu A sau | Phép biến đổi |
|---|------|-------------|-----------|---------------|
| 1 | Upload | file PNG | `PIL.Image` RGB `300×160` | `Image.open` + `verify()` rồi mở lại, kiểm tra content-type và giới hạn 10 MB / 25 M pixel |
| 2 | `crop_margin` | RGB `300×160` | RGB `85×74` | min–max scale về 0–255, nhị phân hóa `pixel < 200` thành mask nét, `cv2.boundingRect` lấy hộp bao, crop đúng vùng công thức — bỏ hết viền trắng |
| 3 | `resize` + `thumbnail` | `85×74` | giữ tỉ lệ, vừa trong `672×192` | resize theo cạnh ngắn `min(192, 672) = 192` rồi `thumbnail((672, 192))` co lại cho không vượt khung; tỉ lệ công thức không bị méo |
| 4 | `ImageOps.expand` | ảnh đã resize | RGB `672×192` | padding **đen** (fill mặc định `0`), chia đều hai bên (`delta // 2`) nên công thức nằm giữa canvas cố định. Mẫu A: 452/672 cột là padding đen, 220 cột giữa là ảnh thật |
| 5 | `ToGray` + `Normalize` | uint8 0–255 | `float32`, shape `[1, 192, 672]` | grayscale, rồi `(x/255 − 0.7931) / 0.1738` theo đúng thống kê UniMERNet; chỉ giữ 1 channel (`[:1]`). Nền trắng 255 → `+1.190`, nét đen 0 → `−4.563`; mean mẫu A = `−2.899`. Lưu ý: padding đen cho ra đúng `−4.563` **trùng giá trị với nét chữ** — model phải tự học phân biệt vùng padding bằng vị trí/hình dạng, đây là thiết kế của UniMERNet chứ không phải lỗi |
| 6 | Batch | `[1, 192, 672]` | `[1, 1, 192, 672]` → `[1, 3, 192, 672]` | `unsqueeze(0)` thêm chiều batch; `DonutEncoderDecoder` tự `repeat(1, 3, 1, 1)` vì encoder khai báo 3 channel |
| 7 | Donut-Swin encoder | `[1, 3, 192, 672]` | `[1, 126, 512]` | patch 4×4 → `48×168` patch; 4 stage (`depths=[6,6,6,6]`, `num_heads=[2,4,8,16]`, window 5) mỗi stage giảm một nửa không gian: `48×168 → 24×84 → 12×42 → 6×21`. `6×21 = 126` vector, mỗi vector 512 chiều. Đây là "chuỗi ảnh" để decoder cross-attention |
| 8 | Token hóa nhãn (chỉ dùng khi train/so sánh) | `\frac{x^2}{y+1}` | 14 id: `[0, 82, 707, 113, 110, 84, 40, 115, 113, 111, 33, 39, 115, 2]` | tokenizer nougat (BPE, vocab 50 000): `<s>`, `\`, `frac`, `{`, `x`, `^`, `2`, `}`, `{`, `y`, `+`, `1`, `}`, `</s>`. `\frac` **bị tách thành 2 token** `\` + `frac` — khác hẳn tokenizer tự viết trong kế hoạch cũ (một token `\frac`). BOS=0, PAD=1, EOS=2 |
| 9 | Decoder (1 bước, teacher forcing) | 13 token đầu vào | logits `[1, 13, 50000]` | mBart-style: `d_model=512`, 8 layer, 16 head, FFN 2048, masked self-attention + cross-attention vào 126 vector của bước 7. Tại vị trí 0, top-1 là token `\` với xác suất **0.9998** (sau đó `{` 0.0002), tức model gần như chắc chắn công thức mở đầu bằng một LaTeX command |
| 10 | Generate thật (greedy) | `[1, 1, 192, 672]` | 24 token id, bắt đầu `[82, 707, 243, 113, 763, ...]` | `model.generate` với `do_sample=False` → greedy, `max_new_tokens=1536`, tự dừng ở EOS. Vòng lặp: encoder chạy 1 lần, decoder sinh token kế tiếp dựa trên toàn bộ token đã sinh (id `243` là token khoảng trắng chen giữa các ký hiệu) |
| 11 | Detokenize | 24 token id | `"\frac { x ^ { 2 } } { y + 1 }"` | `token2str`; **đúng nhãn** của mẫu A, chỉ khác khoảng trắng và `{2}` thay vì `2` — LaTeX tương đương |
| 12 | `TOKEN_PATTERN` (debug) | `pred_str` | 13 token: `\frac { x ^ { 2 } } { y + 1 }` | regex `\\[A-Za-z]+\|\\[^A-Za-z\s]\|[^\s]` gộp lại thành token "người đọc hiểu được": `\frac` là **một** token, khác với cách chia của tokenizer ở bước 8. Chuỗi này chỉ để hiển thị, không đưa lại vào model |
| 13 | Response | — | JSON | `latex` (chuỗi bước 11), `tokens` (bước 12), `latency_ms`, `source_size` = `[300, 160]` (ảnh gốc), `image_size` = `[672, 192]` (canvas bước 4), `input_image` = chính canvas đó dạng PNG base64 — tức debug view hiển thị đúng thứ model nhìn thấy |
| 14 | Browser | JSON | công thức render | `katex.render(latex, {displayMode: true, throwOnError: false})`; textarea giữ mã LaTeX, sửa tới đâu render lại tới đó |

**Độ trễ đo được:** 1034 ms cho mẫu A trên CPU (một lần đo, sau khi model đã nạp sẵn). Đây là một phép đo đơn lẻ, không phải benchmark.

### Vì sao model cần thông tin vị trí

Vẫn là câu hỏi bảo vệ của kế hoạch cũ, nhưng trả lời trên kiến trúc hiện tại: `x^2` và `x_2` có cùng tập nét, chỉ khác **vị trí dọc** của chữ số 2. Nếu 126 vector ở bước 7 không mang thông tin vị trí thì hai ảnh cho cùng một tập vector và model không thể phân biệt. Donut-Swin giải quyết bằng *window attention* cộng *relative position bias* trong từng cửa sổ 5×5 (thay cho positional encoding 2D cộng thẳng vào feature như kế hoạch cũ), nhờ đó mỗi vector vẫn biết nó mô tả vùng nào của ảnh. Phía decoder, positional embedding học được (`max_position_embeddings=1536`) cộng với causal mask đảm bảo token thứ *t* chỉ thấy các token `< t`.

## 5. API

`GET /api/health` → `{"status": "ok"}`

`POST /api/predict` (multipart, field `file`):

| Mã | Khi nào |
|----|---------|
| 415 | content-type không thuộc `image/png`, `image/jpeg`, `image/bmp`, `image/webp` |
| 413 | file > 10 MB, hoặc `width × height` > 25 000 000 pixel |
| 400 | không mở/verify được bằng Pillow (`UnidentifiedImageError`, `OSError`, `DecompressionBombError`), hoặc `prepare_input` trả `None` (ảnh hỏng, rỗng sau khi crop) |
| 200 | `{latex, tokens, latency_ms, source_size, image_size, input_image}` |

`image_size` là canvas `[672, 192]` model nhận, `source_size` là ảnh người dùng upload, `input_image` là canvas grayscale đã crop/resize/pad.

Frontend đã build (`web/dist`) được FastAPI mount tại `/` nếu thư mục tồn tại, nên bản production chạy một process duy nhất.

## 6. Frontend

[web/src/main.ts](web/src/main.ts): chọn file hoặc drag–drop → `POST /api/predict` → render KaTeX, hiển thị `latency_ms`, kích thước ảnh, chuỗi token, và ảnh `input_image`. Biến `activeRequest` tăng theo từng request và kết quả về muộn bị bỏ, nên upload liên tiếp không bị hiển thị chồng kết quả cũ.

## 7. Giới hạn và điểm chưa khớp đã biết

- Debug view hiển thị canvas `672×192` thật (đã sửa, xem mục 13 bảng trên), nhưng là ảnh **trước** bước normalize — không có cách nào xem trực tiếp tensor `[1, 192, 672]` đưa vào encoder.
- Không có số liệu ExpRate / token accuracy / edit distance: repo không chứa tập test và không chạy benchmark nào. Không được trích dẫn con số nào cho phần đánh giá.
- UniMERNet tiny mạnh với công thức in; ảnh mờ, nét viết tay lạ, ma trận lớn hoặc phân số lồng nhiều tầng vẫn sai. Không tuyên bố hỗ trợ đầy đủ ma trận.
- Chỉ greedy decoding (`do_sample=False`). Beam search chưa bật.
- Python 3.10 bắt buộc: stack UniMERNet không cài được trên 3.13.

## 8. Cấu trúc project hiện tại

```text
OCR-Number/
├── README.md
├── Plans.md
├── requirements.txt
├── pytest.ini
├── src/
│   ├── __init__.py
│   └── api.py          # toàn bộ backend: validate, nạp model, predict
├── tests/
│   └── test_api.py     # health, reject non-image, predict với model được monkeypatch
├── web/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/{main.ts,style.css}
├── notebooks/
│   └── notebook36da636c1f.ipynb   # tài liệu nghiên cứu baseline cũ
└── docs/
    ├── mathematical_background.md
    ├── presentation_notes.md
    └── *.docx                      # mô tả pipeline
```

`src/api.py` khoảng 110 dòng và là toàn bộ phần Python chạy thật.

## 9. Những gì đã bỏ so với kế hoạch cũ

| Đã xóa | Vì sao |
|--------|--------|
| `src/preprocess.py` | processor của UniMERNet lo crop/resize/pad/normalize (bước 2–5) |
| `src/tokenizer.py` | dùng tokenizer nougat 50 000 token đi kèm checkpoint |
| `src/model.py` (CNN + TransformerDecoder) | thay bằng Donut-Swin + mBart pretrained |
| `src/dataset.py`, `src/train.py` | không huấn luyện nữa |
| `src/evaluate.py`, `src/inference.py` | không có checkpoint tự train để đánh giá; inference nằm trong `api.py` |
| `tests/test_{preprocess,tokenizer,model,evaluate}.py` | test cho code đã xóa |
| `demo/app.py` (Streamlit) | thay bằng frontend TypeScript + KaTeX |
| `checkpoints/`, `results/`, `assets/` | không còn checkpoint hay file kết quả nào |
| `matplotlib` trong requirements | chỉ dùng cho code đã xóa |

## 10. Kiểm tra

```powershell
.venv\Scripts\python.exe -m pytest -q      # 3 test, không tải model (get_model bị monkeypatch)
```

Test không gọi model thật nên chạy trong ~1 s. Muốn kiểm tra end-to-end thì chạy uvicorn rồi upload ảnh qua giao diện.

## 11. Việc còn lại nếu muốn làm tiếp

1. Hiển thị cả hai cách chia token (tokenizer nougat ở bước 8 vs regex ở bước 12) trong giao diện debug, vì đúng chỗ này dễ bị hỏi khi bảo vệ.
2. Nếu cần số liệu đánh giá: tải một tập test có nhãn, chạy UniMERNet trên đó và báo cáo ExpRate/edit distance — hiện chưa có.
3. Beam search, batch nhiều ảnh, cache theo hash ảnh: chỉ làm khi có nhu cầu thật.
