# Cơ sở toán học

## Từ ảnh đến chuỗi đặc trưng

Ảnh được đổi thành tensor một kênh, trong đó nét mực gần 1 và nền gần 0. CNN học các mẫu cục bộ như cạnh, giao điểm và hình dạng ký hiệu. Feature map `[B, C, H, W]` được trải thành `[B, H×W, C]`, rồi chiếu tuyến tính về `d_model` để decoder có thể cross-attention.

## Vì sao cần vị trí 2D?

Self-attention tự nó không biết thứ tự hay tọa độ. Cùng các nét `x` và `2`, biểu thức `x^2` đặt `2` phía trên còn `x_2` đặt `2` phía dưới. Positional encoding 2D ghép mã sin/cos theo hàng và cột vào từng vector ảnh, nhờ đó mô hình phân biệt được quan hệ không gian. Decoder vẫn dùng vị trí 1D vì LaTeX là một chuỗi có thứ tự.

## Teacher forcing và causal mask

Khi train, decoder nhận chuỗi từ BOS đến token áp chót; target là chuỗi dịch trái một vị trí. Causal mask chặn token hiện tại nhìn các token tương lai. Cross-entropy bỏ qua PAD:

`loss = CrossEntropyLoss(ignore_index=PAD_ID)(logits, target)`

Khi inference, mô hình bắt đầu bằng BOS, chọn token xác suất lớn nhất, nối token đó vào input và dừng ở EOS hoặc giới hạn chiều dài.

## Chỉ số

- **ExpRate:** tỷ lệ toàn bộ chuỗi dự đoán trùng nhãn.
- **Token Accuracy:** `1 - tổng edit distance / tổng độ dài lớn hơn`.
- **Normalized Edit Distance:** trung bình `edit_distance / max(lengths)` theo mẫu; càng thấp càng tốt.
- **Latency:** thời gian greedy decoding trung bình trên một ảnh.

