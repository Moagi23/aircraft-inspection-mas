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

def append_result(row: dict):
    """
    row keys you can pass:
    - experiment_id (str)
    - task (str)
    - agent (str)
    - input_type (str) -> "camera" | "upload"
    - serial_number (str|None)
    - confidence (float)
    - image_path (str|None)
    - notes (str|None)
    """
    _ensure_dirs()
    fieldnames = [
        "timestamp_iso", "experiment_id", "task", "agent", "input_type",
        "serial_number", "confidence", "image_path", "notes"
    ]
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    row_final = {k: row.get(k) for k in fieldnames if k != "timestamp_iso"}
    row_final["timestamp_iso"] = ts

    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row_final)

def read_results() -> list[dict]:
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
