import requests
import numpy as np
from PIL import Image
import io

class SerialNumberAgent:
    def __init__(self, api_url="http://168.119.242.186:8500/scan_serial"):
        self.api_url = api_url

    def scan(self, pil_img):
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        buffered.seek(0)

        files = {'file': ('image.jpg', buffered, 'image/jpeg')}
        try:
            response = requests.post(self.api_url, files=files, timeout=10)
            if response.status_code == 200:
                data = response.json()
                serial_number = data.get("serial_number")
                confidence = data.get("confidence", 0.0)
                return serial_number, confidence
            else:
                print("API call failed:", response.text)
        except Exception as e:
            print("Error calling OCR API:", e)

        return None, 0.0
