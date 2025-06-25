import re
import numpy as np
from PIL import Image

class SerialNumberAgent:
    def __init__(self):
        self.ocr = None

    def scan(self, pil_img):
        if self.ocr is None:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        img = np.array(pil_img)
        results = self.ocr.ocr(img)
        serial_keywords = ['SER', 'SERNO', 'SER NO', 'SERIAL', 'S/N']
        serial_number = None
        serial_conf = 0.0

        if results and results[0]:
            boxes = results[0]
            for i, (box, (text, conf)) in enumerate(boxes):
                print(f"[{i:02}] Text: '{text}' | Confidence: {conf:.2f}")

            for i, (box, (text, conf)) in enumerate(boxes):
                upper_text = text.upper().strip()
                for kw in serial_keywords:
                    if upper_text.startswith(kw):
                        match = re.search(rf"{kw}\s+([A-Z0-9\-]+)", upper_text)
                        if match:
                            serial_number = match.group(1)
                            serial_conf = conf
                            break
                if serial_number:
                    break

            if not serial_number:
                for i, (box, (text, conf)) in enumerate(boxes):
                    clean_text = text.upper().replace(" ", "")
                    if any(k in clean_text for k in serial_keywords):
                        x0, y0 = box[0]
                        min_dist = float('inf')
                        best_match = None

                        for j, (other_box, (other_text, other_conf)) in enumerate(boxes):
                            if j == i:
                                continue
                            x1, y1 = other_box[0]
                            if abs(y1 - y0) < 20 and x1 > x0:
                                dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    best_match = other_text
                                    serial_conf = other_conf

                        if best_match:
                            serial_number = best_match.strip()
                            break

        if serial_number:
            print(serial_number, serial_conf)
            return serial_number, serial_conf
        else:
            return None, 0.0
