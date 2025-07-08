import requests
import io
import base64
from PIL import Image
import openai
import os
import streamlit as st
from dotenv import load_dotenv


try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except st.runtime.secrets.StreamlitSecretNotFoundError:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

class SerialNumberAgent:
    def __init__(self, api_url="http://168.119.242.186:8500/scan_serial"):
        self.api_url = api_url

    def scan(self, pil_img):
        serial_number, confidence = self._try_api(pil_img)

        if confidence >= 0.8:
            print(f"✅ Using OCR API result: {serial_number} (Confidence: {confidence})")
            return serial_number, confidence
        else:
            print(f"⚠️ OCR confidence too low ({confidence}). Falling back to GPT...")
            gpt_serial = self._use_gpt(pil_img)
            return gpt_serial, confidence  # or maybe confidence = 1.0 for GPT fallback


    def _try_api(self, pil_img):
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

    def _use_gpt(self, pil_img):
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = (
            "Extract the serial number from this image. "
            "It may be labeled as 'Serial No.', 'SER', 'Snr', etc. "
            "Return only the serial number value."
        )

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Error calling GPT-4o:", e)
            return None
