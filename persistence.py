# persistence.py
import os, csv, uuid, datetime
from PIL import Image

RESULTS_DIR = "results"
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
CSV_PATH = os.path.join(RESULTS_DIR, "experiments.csv")


def _ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def save_image(pil_img) -> str:
    """Save image and return relative path."""
    _ensure_dirs()
    img_name = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join(IMAGES_DIR, img_name)
    pil_img.save(img_path, format="JPEG", quality=92)
    return os.path.relpath(img_path)


# --- new helpers for dynamic CSV schema ---
def _read_csv():
    if not os.path.exists(CSV_PATH):
        return [], []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return (reader.fieldnames or []), list(reader)


def _write_csv(fieldnames, rows):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # fill missing keys with empty string
            w.writerow({k: r.get(k, "") for k in fieldnames})


def append_result(row: dict):
    """
    Append a row to experiments.csv.
    Automatically expands the CSV header if new keys appear (e.g., ts_scan_pressed).
    """
    _ensure_dirs()

    # keep your existing timestamp_iso behavior
    row = dict(row)  # copy to avoid side effects
    row.setdefault("timestamp_iso", datetime.datetime.now().isoformat(timespec="seconds"))

    existing_fields, rows = _read_csv()

    # default field order (keeps your old layout, but allows new cols after)
    default_fields = [
        "timestamp_iso", "experiment_id", "case_id",
        "task", "agent", "input_type",
        "serial_number", "confidence",
        "image_path", "notes",
        # timeline fields
        "ts_camera_start", "ts_scan_pressed", "ts_ocr_result",
        "ts_gpt_result", "ts_gpt_verification",
        "ts_accept_save_pressed", "ts_edit_pressed", "ts_save_edited_pressed",
        "ts_result_saved",
    ]


    if not existing_fields:
        # brand new file: start with default_fields + any extra keys
        fieldnames = list(default_fields)
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
        _write_csv(fieldnames, [])
    else:
        # existing file: expand header if needed
        fieldnames = list(existing_fields)
        changed = False
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
                changed = True
        if changed:
            # rewrite old rows with the expanded header
            _write_csv(fieldnames, rows)

    # append the new row with whatever columns exist
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writerow({k: row.get(k, "") for k in fieldnames})


def read_results() -> list[dict]:
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
