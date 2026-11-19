"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
import math

from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401

CHECK = "label_format"


def _stems(ds):
    """Tên (stem) của mọi ảnh trong dataset.
    Nếu dòng nào dưới đây không khớp với common.py của bạn thì sửa lại ở đây."""
    for attr in ("stems", "images", "image_stems"):
        v = getattr(ds, attr, None)
        if v is not None:
            v = v() if callable(v) else v
            return sorted(getattr(s, "stem", s) for s in v)
    return sorted(p.stem for p in ds.labels_dir.glob("*.txt"))


def run(ds, cfg):
    c = cfg["label_format"]
    tol = c["overflow_tol"]
    dup_iou = c["dup_box_iou"]
    min_px = c["min_box_px"]
    max_frac = c["max_box_frac"]
    n_cls = len(ds.classes)

    issues = []

    def add(sev, code, stem, msg, line, **extra):
        issues.append(Issue(check=CHECK, severity=sev, code=code, image=stem,
                            message=msg, line=line, extra=extra))

    for stem in _stems(ds):
        path = ds.label_path(stem)
        if not path.exists():
            continue  # thiếu file nhãn là việc của check missing_labels

        size = None          # lấy kích thước ảnh khi cần, đúng 1 lần
        seen = []            # (cls, x, y, w, h) của các dòng hợp lệ trước đó

        for ln, raw in enumerate(path.read_text().splitlines(), start=1):
            if not raw.strip():
                continue     # dòng trống: bỏ qua nhưng vẫn đếm số dòng

            # 1) định dạng
            try:
                cls, x, y, w, h = parse_yolo_line(raw)
                cls = int(cls)
                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError("nan/inf")
            except (ValueError, TypeError):
                add(ERROR, "BAD_FORMAT", stem, f"dòng không đúng 5 cột số: {raw!r}", ln)
                continue

            # 2) id lớp
            if not 0 <= cls < n_cls:
                add(ERROR, "BAD_CLASS_ID", stem, f"id lớp {cls} ngoài 0..{n_cls - 1}", ln)
                continue

            # 3) box rỗng
            if w <= 0 or h <= 0:
                add(ERROR, "BOX_ZERO", stem, f"box có w={w}, h={h}", ln)
                continue

            # 4) tràn ảnh: mức tràn lớn nhất trong 4 mép
            overflow = max(-(x - w / 2), (x + w / 2) - 1,
                           -(y - h / 2), (y + h / 2) - 1)
            if overflow > tol:
                add(ERROR, "BOX_OUT_OF_IMAGE", stem,
                    f"box tràn ra ngoài ảnh {overflow:.4f}", ln, overflow=overflow)

            # 5) trùng: cùng lớp, IoU > ngưỡng với một dòng TRƯỚC, báo ở dòng sau
            for c2, x2, y2, w2, h2 in seen:
                if c2 == cls and iou_xywh((x, y, w, h), (x2, y2, w2, h2)) > dup_iou:
                    add(ERROR, "DUP_BOX", stem, "box trùng một dòng phía trước", ln)
                    break
            seen.append((cls, x, y, w, h))

            # 6) quá nhỏ (cần kích thước ảnh)
            if size is None:
                try:
                    size = ds.image_size(stem)
                except Exception:
                    size = False
            if size and min(w * size[0], h * size[1]) < min_px:
                add(WARN, "BOX_TINY", stem, "box quá nhỏ", ln)

            # 7) quá lớn
            if w * h > max_frac:
                add(WARN, "BOX_HUGE", stem, "box chiếm gần hết ảnh", ln)

    return issues