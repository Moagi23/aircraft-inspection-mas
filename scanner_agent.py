from PIL import Image
from serial_number_agent import SerialNumberAgent

class ScannerAgent:
    """
    Always-run scanner pipeline:
      OCR -> GPT extraction -> GPT verification
    Intended for 'scan scan scan' workflow (auto-save in UI).
    """

    is_auto_save = True  # UI can use this to auto-save and hide review/edit

    def __init__(self, api_url: str = "http://168.119.242.186:8500/scan_serial",
                 ocr_early_accept_threshold: float = 0.95,
                 min_ocr_conf_to_save: float | None = None):
        # Reuse the exact logic from SerialNumberAgent
        self.base = SerialNumberAgent(
            api_url=api_url,
            ocr_early_accept_threshold=ocr_early_accept_threshold,
        )
        self.min_ocr_conf_to_save = min_ocr_conf_to_save

    def scan(self, pil_img: Image.Image):
        """
        Returns (serial_number, ocr_conf).
        If min_ocr_conf_to_save is set and OCR confidence is below it, returns (None, ocr_conf).
        """
        serial_number, ocr_conf = self.base.scan(pil_img)

        # Optional: don't save very weak OCR cases
        if self.min_ocr_conf_to_save is not None:
            if ocr_conf is None:
                return None, ocr_conf
            if float(ocr_conf) < float(self.min_ocr_conf_to_save):
                return None, float(ocr_conf)

        return serial_number, ocr_conf
