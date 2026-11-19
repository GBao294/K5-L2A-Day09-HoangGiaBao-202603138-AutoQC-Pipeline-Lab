"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def _stem(f):
    return getattr(f, "stem", f)


def _norm(box):
    """Chuẩn hoá (id, cx, cy, w, h) để so sánh hai nguồn parse."""
    return (int(box[0]),) + tuple(float(v) for v in box[1:5])


def run(ds, cfg, preds=None):
    c = cfg["second_opinion"]
    conf_min, min_area = c["conf"], c["min_area_px"]
    fp_conf = c["fp_conf"]
    iou_missing, iou_match, iou_loose = c["iou_missing"], c["iou_match"], c["iou_loose"]

    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    issues = []

    def add(code, stem, msg, line=None, **extra):
        issues.append(Issue(check="second_opinion", severity=WARN, code=code,
                            image=stem, message=msg, line=line, extra=extra))

    for f in ds.image_files():
        stem = _stem(f)
        if stem not in preds or not ds.label_path(stem).exists():
            continue                       # không có dự đoán hoặc chưa có nhãn: bỏ qua

        W, H = ds.image_size(stem)
        L = [_norm(b) for b in ds.valid_boxes(stem)]      # nhãn hợp lệ (đã bỏ dòng hỏng)
        all_preds = preds[stem]

        # --- A) mỗi dự đoán đủ tin cậy và đủ lớn: MISSING / LOOSE_BOX / WRONG_CLASS ---
        P = [r for r in all_preds
             if r[1] >= conf_min and (r[4] * W) * (r[5] * H) >= min_area]
        for r in P:
            pcls, pconf = int(r[0]), r[1]
            pbox = tuple(r[2:6])
            cands = []                                    # (iou, lớp nhãn) của MỌI lớp, IoU >= iou_missing
            for lb in L:
                iou = iou_xywh(pbox, lb[1:])
                if iou >= iou_missing:
                    cands.append((iou, lb[0]))

            if not cands:                                 # (1) model thấy vật mà nhãn không có box
                add("POSSIBLE_MISSING", stem,
                    f"model thấy lớp {pcls} (conf {pconf:.2f}) nhưng nhãn không có box",
                    conf=pconf)
                continue

            same = [iou for iou, cl in cands if cl == pcls]
            other = [iou for iou, cl in cands if cl != pcls]
            if same:                                      # (2) nhãn đúng lớp đã phủ vật này
                best = max(same)
                if best < iou_loose:
                    add("POSSIBLE_LOOSE_BOX", stem,
                        f"box đúng lớp nhưng lỏng/lệch (IoU {best:.2f})",
                        conf=pconf, iou=best)
            elif other and max(other) >= iou_match:       # (3) chỉ có nhãn khác lớp, khớp rõ
                best = max(other)
                add("POSSIBLE_WRONG_CLASS", stem,
                    f"model đoán lớp {pcls} nhưng nhãn gán lớp khác (IoU {best:.2f})",
                    conf=pconf, iou=best)
            # còn lại: khác lớp, IoU trong [iou_missing, iou_match) -> mơ hồ, không cờ

        # --- B) mỗi dòng nhãn hợp lệ mà không khớp dự đoán nào: FALSE_POS (có số dòng) ---
        fp_preds = [tuple(r[2:6]) for r in all_preds if r[1] >= fp_conf]   # không lọc diện tích
        valid = set(L)
        for item in ds.read_label_lines(stem):
            # item là (số_dòng, nội_dung) hoặc (nội_dung, số_dòng): nhận cả hai thứ tự
            a, b = item
            ln, raw = (a, b) if isinstance(a, int) else (b, a)
            try:
                box = _norm(parse_yolo_line(raw))
            except (ValueError, TypeError, AttributeError):
                continue                                  # dòng hỏng/trống: bỏ qua, vẫn giữ số dòng gốc
            if box not in valid:
                continue
            best = max((iou_xywh(p, box[1:]) for p in fp_preds), default=0.0)
            if best < iou_missing:
                add("POSSIBLE_FALSE_POS", stem,
                    "box nhãn không khớp vật nào model thấy", line=ln, iou=best)

    return issues