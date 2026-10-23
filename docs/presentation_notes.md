# Ghi chú bảo vệ

1. Mở bằng bài toán: OCR ký tự đơn không đủ vì công thức có cấu trúc 2D.
2. Cho xem đúng một mẫu CROHME trong `02_single_sample_walkthrough.ipynb`.
3. Giải thích preprocessing làm giảm vùng nền nhưng không kéo méo tỉ lệ.
4. Theo tensor shape thật: ảnh → feature map CNN → chuỗi memory → logits token.
5. So sánh `x^2` với `x_2` để giải thích vị trí 2D.
6. Minh họa shift BOS/input và target/EOS, causal mask và một giá trị loss thật.
7. Báo cáo ExpRate, token accuracy, edit similarity và latency từ file JSON do chương trình sinh.
8. Trình bày lỗi thật theo nhóm: ký hiệu, mũ/chỉ số, phân số, ngoặc, thiếu/thừa token.
9. Nói rõ giới hạn coverage, đặc biệt với ma trận; không suy rộng từ mẫu chưa đánh giá.

Checklist trước demo: checkpoint tồn tại, ảnh thử thuộc test split, chạy inference CPU trước, giữ ảnh và LaTeX nhãn để đối chiếu, không chỉnh tay kết quả dự đoán khi báo cáo metric.

