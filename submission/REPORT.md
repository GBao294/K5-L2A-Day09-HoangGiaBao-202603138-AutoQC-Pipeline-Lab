# Báo cáo Day 09 — Hoàng Gia Bảo, 202603138

## 1. Kết luận trước
Pipeline kết luận **Release**. **224 ảnh vào**; bước 1 loại 31 ảnh (8 trùng byte, 7 trùng gần đúng, 11 rò rỉ test/val, 3 quá nhỏ, 2 hỏng) nên **193 ảnh ra dataset** (`submission/export/clean_v1`). Lý do chính: sau sửa tự động chỉ còn 6 lỗi bắt buộc (6 ảnh chưa có nhãn); người duyệt (mô phỏng, ngân sách 80 ảnh) xử lý xong nên n_errors = 0, và 13/193 ảnh (6,74%) còn cảnh báo, thấp hơn ngưỡng Hold 10%. Cùng lô nhưng ngân sách 0 thì ra Rework (mục 4).

## 2. Lỗi và phân việc
Nguồn: `02_error_list_v1`. Bước 1 loại 31 ảnh; nhãn của 193 ảnh còn lại có 36 lỗi bắt buộc và 108 cảnh báo.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| LEAK_TEST | 11 ảnh | máy (loại ảnh) | trùng ảnh kho test/val, giữ lại sẽ làm mAP bị thổi phồng |
| DUP_IMAGE_EXACT | 8 ảnh | máy (loại ảnh) | trùng từng byte, không cần người phán đoán |
| DUP_IMAGE_NEAR | 7 ảnh | máy (loại ảnh) | trùng gần đúng theo phash (xem mục 3) |
| BAD_SIZE / CORRUPT_IMAGE | 3 / 2 ảnh | máy (loại ảnh) | cạnh ngắn < 250px / không mở được |
| BAD_CLASS_ID | 6 dòng | máy (FIXED_DROP_LINE) | id lớp ngoài 0..3, bỏ dòng |
| BAD_FORMAT | 3 dòng | máy (FIXED_DROP_LINE) | có `nan` hoặc thiếu cột |
| BOX_ZERO | 4 dòng | máy (FIXED_DROP_LINE) | w hoặc h = 0 |
| DUP_BOX | 7 dòng | máy (FIXED_DROP_LINE) | box trùng dòng phía trước |
| BOX_OUT_OF_IMAGE | 8 dòng | máy (FIXED_CLIP_BOX) | tràn 0,023–0,06, lớn hơn `overflow_tol` 0,002, xén về trong ảnh |
| BAD_CLASS_NAME | 2 dòng (data.yaml) | máy | tên 'Person', 'Bus' sai chuẩn, chuẩn hoá lại |
| MISSING_LABEL_FILE | 6 ảnh | người (NEEDS_LABEL) | máy không tự tạo nhãn từ đầu; đây chính là 6 lỗi còn lại sau vòng 2 |
| POSSIBLE_MISSING | 32 ảnh | người | model lớn thấy vật mà nhãn không có; 26/32 là lớp person |
| POSSIBLE_FALSE_POS | 28 dòng / 26 ảnh | người | box nhãn không khớp vật nào model thấy |
| POSSIBLE_WRONG_CLASS | 13 ảnh | người | cùng vị trí (IoU 0,77–0,99) nhưng khác lớp |
| POSSIBLE_LOOSE_BOX | 11 ảnh | người | đúng lớp nhưng IoU 0,34–0,48 |
| EMPTY_LABEL | 11 ảnh | người | không phân biệt được ảnh nền với thiếu nhãn |
| BOX_TINY / BOX_HUGE / CLASS_RARE | 6 dòng (4 ảnh) / 3 / 4 dòng | người nếu còn ngân sách | trọng số thấp (0,3–0,5) |

**Vì sao `05_review_queue.csv` xếp như vậy:** priority là tổng trọng số trong `pipeline.yaml`; với mã do model đề xuất thì nhân thêm (1 + conf) (số liệu khớp: img_0028 có MISSING conf 0,94 → 2 × 1,94 ≈ 3,881). Vì thế 6 ảnh NEEDS_LABEL đứng đầu với 10,0 (thiếu nhãn thì không xuất được). img_0020 hạng 1 (11,524) vì cộng dồn WRONG_CLASS + MISSING + DROP_LINE. Tiếp theo là WRONG_CLASS (trọng số 3), rồi MISSING/FALSE_POS (2), LOOSE_BOX (1,5); BOX_TINY/BOX_HUGE xuống cuối.

**Ngân sách 80:** hàng đợi có 98 ảnh nên người xem được **67/69 ảnh có lỗi ngữ nghĩa (97,1%)**: 32/32 MISSING, 13/13 WRONG_CLASS, 11/11 LOOSE_BOX, 20/22 FALSE_POS. Hàng đợi dựng sau sửa tự động nên chỉ còn 22/26 ảnh FALSE_POS (img_0050, 0066, 0185, 0108 đã bị bỏ dòng hoặc xén box). Hai ảnh bị bỏ là img_0189 và img_0206 (mỗi ảnh 1 FALSE_POS). Điểm yếu: vạch 80 cắt giữa nhóm 17 ảnh cùng priority 2,0 (hạng 67–83). Trong 14 ảnh được vào có 6 ảnh chỉ có FIXED_DROP_LINE (máy đã sửa), trong khi 2 ảnh FALSE_POS bị loại.

## 3. Calibration (2 ngưỡng)
Nguồn: `submission/calibration/calibration_report.md` (bộ calib có chèn lỗi biết trước). P = precision, R = recall.

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 | DUP_IMAGE_NEAR: P 1,00 → 1,00; R 0,50 → 0,83 (F1 0,67 → 0,91). DUP_IMAGE_EXACT không đổi (1,00 / 1,00) | Ưu tiên loại ảnh dư thừa không có ý nghĩa, nên lấy ngưỡng có recall cao nhất mà precision vẫn 1,00. Ngưỡng 20 thì sụp (NEAR R 0,50; EXACT R 0,33) |
| `second_opinion.conf` | 0.25 | 0.8 | WRONG_CLASS: P 0,56 → 1,00, R 0,83 giữ nguyên. MISSING: P 0,50 → 0,67, R 1,00 → 0,75. LOOSE_BOX: P 0,29 → 0,50, R 0,40 → 0,20. FALSE_POS: 0,31 / 1,00 không đổi | WRONG_CLASS có trọng số hàng đợi cao nhất (3), conf 0,8 loại hết 4 cảnh báo sai (F1 0,67 → 0,91) nên ngân sách 80 ảnh không bị phí; F1 trung bình 4 mã ≈ 0,60 so với ≈ 0,54 ở 0,25. Đổi lại MISSING mất 2/8 hộp thiếu |

**Chỗ calibration có thể đánh lừa:** precision của DUP_IMAGE_NEAR luôn 1,00 ở mọi ngưỡng, kể cả 20, nghĩa là bộ calib không có ảnh "giống cảnh nhưng khác ảnh" để phạt việc nới ngưỡng. Lợi ích của 16 chỉ là thêm 1 cặp trong 6 (R 0,67 → 0,83) và nó đứng sát vách đá: ở 20 recall tụt về 0,50. Trên dữ liệu thật, 3/7 cặp NEAR nằm đúng ở khoảng cách 16 (img_0011–0008, 0084–0074, 0177–0113; 4 cặp còn lại ở 0, 0, 6, 10). Nếu cả 3 cặp là nhầm thì chỉ mất 3/224 ảnh (1,3%), rủi ro có chặn nhưng chưa được chứng minh: ⟦CẦN: kết quả mở xem 3 cặp ảnh, trùng thật hay chỉ giống cảnh⟧. Riêng img_0113 đã bị loại sẵn vì cạnh ngắn 226px (< 250), nên cặp 0177–0113 mất cả hai ảnh. Với `conf` cũng tương tự: một ngưỡng chung cho nhiều mã có điểm tối ưu khác nhau (MISSING tốt nhất ở 0,5–0,6 với F1 0,73, ở 0,8 là 0,71; LOOSE_BOX tốt nhất ở 0,1 với F1 0,46, ở 0,8 là 0,29), và nâng `conf` làm số cảnh báo giảm nên cửa chặn cũng dễ qua hơn.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Cửa chặn.** Lệnh `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt) cho:

```text
[09:45:31] Bước 3/5  Người duyệt (MÔ PHỎNG) — ngân sách 0 ảnh
[09:45:33]           kết luận: REWORK — Còn 6 lỗi bắt buộc chưa sửa: phải làm lại (Rework).
[09:45:33] Bước 5/5  KHÔNG xuất: còn lỗi bắt buộc (Rework) — trả về bước 3
```

Cùng dữ liệu, chạy mặc định (ngân sách 80) cho **Release** (13/193 ảnh còn cảnh báo, 6,74% < 10%) và xuất `submission\export\clean_v1`. Khác biệt duy nhất là 6 ảnh chưa có nhãn được người duyệt gán.

**(b) Điểm mù.** Số liệu lấy từ bộ calib (cấu hình đã chọn: phash 16, conf 0,8), không phải từ dữ liệu thật:
1. **Box lỏng nhẹ:** LOOSE_BOX chỉ có R 0,20 (bắt 1/5 hộp lỏng được chèn), P 0,50. Luật `iou_loose` = 0,5 nên hộp lỏng nhưng IoU ≥ 0,5 không bị báo; cả 11 ảnh được báo trên dữ liệu thật đều có IoU 0,34–0,48.
2. **Vật thiếu nhãn mà model lớn thấy với conf thấp:** MISSING R 0,75 (sót 2/8) ở conf 0,8, trong khi ở conf ≤ 0,6 là 1,00. Cả 32 cảnh báo MISSING trên dữ liệu thật đều có conf 0,82–0,96, nên vật có conf dưới 0,8 không bao giờ được báo.
3. **Nhầm lớp mà model lớn cũng nhầm:** WRONG_CLASS giữ R 0,83 ở mọi conf từ 0,1 đến 0,8, gợi ý 1/6 hộp sai lớp mà hạ ngưỡng cũng không bắt được.
4. **Ảnh trùng biến đổi mạnh:** NEAR R 0,83 (sót 1/6) và không ngưỡng nào trong 2–20 đạt 1,00.

Ngược lại, FALSE_POS có P 0,31 (11 báo sai trên 16 lần báo) mà `conf` không làm thay đổi, nên nhiều slot trong ngân sách 80 ảnh có thể bị tiêu vào cảnh báo nhiễu.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy. 

## 6. Quyết định và ghi chú AI
### 6.1. Quyết định cuối
Giữ kết luận **Release** và xuất `clean_v1` (193 ảnh). Ưu tiên của tôi là loại trước các ảnh không có ý nghĩa (trùng, rò rỉ test), nên nới `phash_dist` lên 16 (NEAR R 0,83, P 1,00 trên bộ calib) và giữ `conf` 0,8 để hàng đợi 80 ảnh không bị lấp bởi cảnh báo sai lớp. Rủi ro chấp nhận: trên bộ calib LOOSE_BOX chỉ bắt được 1/5, MISSING sót 2/8; trên dữ liệu thật còn 13 ảnh cảnh báo chưa ai duyệt (6,74% < 10%). Nếu có thêm ngân sách người duyệt, tôi sẽ hạ `conf` về 0,5–0,6 để lấy lại recall MISSING (1,00).

### 6.2. Nhờ AI viết script (mô tả đủ 4 ý)

Khi nhờ AI viết script kiểm tra nhãn YOLO, cần mô tả rõ 4 nội dung:

1. **Định dạng đầu vào:** Thư mục `labels/`, các file `.txt` theo định dạng YOLO. Mỗi dòng gồm 5 cột: `class_id`, `x_center`, `y_center`, `width`, `height`; các giá trị tọa độ và kích thước được chuẩn hóa trong khoảng 0–1.
2. **Quy tắc kiểm tra:** Kiểm tra bounding box có nằm ngoài phạm vi ảnh hay không.
3. **Kiểu đầu ra:** In tên file, số dòng và lý do phát hiện lỗi.
4. **Giới hạn an toàn:** Chỉ đọc file và thông báo lỗi, không chỉnh sửa hoặc xóa dữ liệu.

Ví dụ prompt:

```text
Viết script Python 3 kiểm thư mục labels/ (YOLO txt).
Mỗi dòng: id x_tam y_tam rong cao, tỉ lệ 0-1.
Ví dụ: 1 0.50 0.50 0.20 0.30
In "tên file, số dòng, lý do" cho dòng có box tràn ảnh.
Chỉ đọc file, không sửa, không xóa. Dưới 12 dòng.
Giải thích từng dòng bằng tiếng Việt.
```

Một prompt rõ ràng giúp AI hiểu đúng đầu vào, quy tắc kiểm tra, kết quả cần in và giới hạn thao tác. Ví dụ, thay vì chỉ yêu cầu "viết code kiểm nhãn", cần nói rõ chỉ kiểm tra box tràn ảnh và không được sửa file.

**Áp dụng trong lab:** Các file trong `pipeline/checks/` đã có docstring mô tả đầu vào và đầu ra. Tôi sử dụng docstring cùng với ít nhất một test liên quan làm thông tin đầu vào cho AI, thay vì tự mô tả lại toàn bộ yêu cầu. Khi cần kiểm tra dữ liệu, tôi chỉ cung cấp một số dòng nhãn mẫu, không gửi ảnh chứa thông tin nhạy cảm như khuôn mặt hoặc biển số.

### 6.3. Nhật ký sử dụng AI

Tôi sử dụng AI để hỗ trợ sửa code trong 4 module. Đầu vào gồm file ghi log lỗi pytest và hướng dẫn của lab (docs/PROMPTING.md).
AI hỗ trợ: Đọc lỗi pytest, xác định nguyên nhân và đề xuất chỉnh sửa code.
Tôi kiểm tra: Đối chiếu với code hiện có, chạy lại pytest sau mỗi lần sửa và kiểm tra kết quả.
AI gặp lỗi: AI từng dùng sai mã DUPLICATE_BOX thay vì DUP_BOX, thiếu import trong second_opinion.py và giả định sai kiểu dữ liệu trả về của ds.read_label_lines. Tôi dựa vào traceback và code trong project để yêu cầu AI sửa lại.
**Kinh nghiệm rút ra:** AI giúp triển khai code nhanh hơn nhưng không phải lúc nào cũng hiểu đúng cấu trúc và kiểu dữ liệu của project. Vì vậy, cần cung cấp docstring, test và thông tin từ các module liên quan; sau đó phải tự chạy test, đọc traceback và kiểm tra lại những thay đổi do AI đề xuất.
