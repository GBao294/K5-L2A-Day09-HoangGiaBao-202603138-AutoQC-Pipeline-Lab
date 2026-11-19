"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _is_remove(code):
    return code in REMOVE_CODES or code.startswith("DUP_IMAGE")


def _yaml_path(ds):
    for attr in ("yaml_path", "data_yaml"):
        p = getattr(ds, attr, None)
        if p is not None:
            return p() if callable(p) else p
    return ds.labels_dir.parent / "data.yaml"


def _image_files(ds, stem):
    """Mọi file ảnh có tên `stem` (kể cả đuôi .JPG)."""
    if hasattr(ds, "image_path"):
        try:
            p = ds.image_path(stem)
            return [p] if p.exists() else []
        except Exception:
            pass
    d = getattr(ds, "images_dir", ds.labels_dir.parent / "images")
    return [p for p in d.glob("*") if p.stem == stem]


def _remove_image(ds, stem):
    if hasattr(ds, "remove_image"):          # nếu Dataset đã có sẵn thì dùng
        ds.remove_image(stem)
        return
    for p in _image_files(ds, stem):
        p.unlink()
    lp = ds.label_path(stem)
    if lp.exists():
        lp.unlink()


def _clip(raw):
    """Cắt box về [0,1]. Trả về dòng mới, hoặc None nếu box không còn diện tích."""
    cls, x, y, w, h = parse_yolo_line(raw)
    l, r = max(0.0, x - w / 2), min(1.0, x + w / 2)
    t, b = max(0.0, y - h / 2), min(1.0, y + h / 2)
    nw, nh = r - l, b - t
    if nw <= 0 or nh <= 0:
        return None
    return f"{int(cls)} {(l + r) / 2:.6f} {(t + b) / 2:.6f} {nw:.6f} {nh:.6f}"


def fix(ds, issues, cfg):
    fixes = []
    errs = [i for i in issues if i.severity == ERROR]   # WARN không bao giờ đụng tới

    def rec(image, line, action, before, after):
        fixes.append({"image": image, "line": line, "action": action,
                      "before": before, "after": after})

    # 1) Xoá ảnh trước; mọi issue khác của ảnh đó sau này bị bỏ qua êm
    removed = set()
    for i in errs:
        if _is_remove(i.code) and i.image not in removed:
            _remove_image(ds, i.image)
            removed.add(i.image)
            rec(i.image, None, "remove_image", i.code, "(đã xoá ảnh và nhãn)")

    # 2) Gom lỗi theo ảnh để xử lý đúng MỘT lần mỗi file nhãn
    by_image = defaultdict(list)
    for i in errs:
        if i.image in removed or _is_remove(i.code):
            continue
        by_image[i.image].append(i)

    for stem, lst in by_image.items():
        # needs_label: không tạo file, chỉ ghi log (ghi lại mỗi lần còn lỗi)
        if any(i.code == "MISSING_LABEL_FILE" for i in lst):
            rec(stem, None, "needs_label", "(không có file)", "chuyển người gán")

        # data.yaml: ghi lại names đúng cfg["classes"]
        if any(i.code == "BAD_CLASS_NAME" for i in lst):
            yp = _yaml_path(ds)
            data = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
            old = data.get("names")
            new = ({k: n for k, n in enumerate(cfg["classes"])}
                   if isinstance(old, dict) else list(cfg["classes"]))
            if old != new:
                data["names"] = new
                yp.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                              encoding="utf-8")
                rec(stem, None, "rename_classes", str(old), str(new))

        # Sửa dòng: dùng số dòng GỐC, ghi lại một lần
        line_codes = defaultdict(set)
        for i in lst:
            if i.line is not None and i.code in DROP_CODES | {"BOX_OUT_OF_IMAGE"}:
                line_codes[i.line].add(i.code)
        if not line_codes:
            continue
        path = ds.label_path(stem)
        if not path.exists():
            continue
        lines = path.read_text().splitlines()      # giữ cả dòng trống để không lệch số dòng
        out = list(lines)
        drop = set()
        for ln in sorted(line_codes):
            if not 1 <= ln <= len(lines):
                continue
            raw = lines[ln - 1]
            codes = line_codes[ln]
            if codes & DROP_CODES:                 # xoá thắng cắt: chỉ ghi 1 fix drop_line
                drop.add(ln)
                rec(stem, ln, "drop_line", raw, "(đã xoá dòng)")
                continue
            try:
                new = _clip(raw)
            except (ValueError, TypeError):
                new = None
            if new is None:                        # box nằm hẳn ngoài ảnh: xoá, không ghi box rỗng
                drop.add(ln)
                rec(stem, ln, "drop_line", raw, "(đã xoá dòng)")
            else:
                out[ln - 1] = new
                rec(stem, ln, "clip_box", raw, new)
        kept = [t for n, t in enumerate(out, start=1) if n not in drop]
        path.write_text("\n".join(kept) + ("\n" if kept else ""))

    return fixes