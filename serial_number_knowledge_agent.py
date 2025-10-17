import io
import base64
import requests
from PIL import Image
import openai
import os
import streamlit as st
from dotenv import load_dotenv

from knowledge_agent import KnowledgeAgent  # ✅ import your KnowledgeAgent

try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except st.runtime.secrets.StreamlitSecretNotFoundError:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


class SerialNumberKnowledgeAgent:
    """
    A hybrid agent combining OCR, GPT verification, and KnowledgeAgent checking.
    """

    def __init__(self, api_url="http://168.119.242.186:8500/scan_serial"):
        self.api_url = api_url
        self.knowledge_agent = KnowledgeAgent()

    def scan(self, pil_img):
        print("🔍 Starting knowledge-based serial number scan...")

        # Step 1: Try PaddleOCR API
        ocr_serial, ocr_conf = self._try_api(pil_img)
        if ocr_serial:
            print(f"📄 OCR result: {ocr_serial} (Confidence: {ocr_conf:.2f})")

            # Step 2: Check in KnowledgeAgent
            if ocr_serial in self.knowledge_agent.get_important_serials():
                print("✅ Found in KnowledgeAgent after OCR. Auto-accepting.")
                return ocr_serial, 1.0

        # Step 3: Use GPT if not found
        gpt_serial = self._use_gpt(pil_img)
        if gpt_serial:
            print(f"🤖 GPT-4o result: {gpt_serial}")

            # Step 4: Check again in KnowledgeAgent
            if gpt_serial in self.knowledge_agent.get_important_serials():
                print("✅ Found in KnowledgeAgent after GPT. Auto-accepting.")
                return gpt_serial, 1.0

        # Step 5: Fallback verification like SerialNumberAgent
        verified = self._verify_with_gpt(pil_img, ocr_serial, gpt_serial)
        if verified:
            print(f"🧪 Verified serial number: {verified}")
            return verified, max(ocr_conf, 0.8)
        else:
            print("⚠️ No reliable serial number detected.")
            return None, 0.0

    # --- Internal helpers ---

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
                print(f"❌ OCR API error: {response.status_code} - {response.text}")
                return None, None
        except Exception as e:
            print("❌ OCR API not reachable:", e)
            return None, None

    def _use_gpt(self, pil_img):
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = (
            "Extract the serial number from this image. "
            "It may be labeled as SER', 'SERNO', 'SER NO', 'SERIAL', 'S/N', 'ESN', etc. "
            "Ignore values labeled MODEL, TYPE, PNR, CERT, DATE, EXP, or MFR. "
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
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            },
                        ],
                    }
                ],
                max_tokens=50,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Error calling GPT-4o:", e)
            return None

    def _verify_with_gpt(self, pil_img, ocr_serial, gpt_serial):
        print("🧪 Verifying results with GPT-4o...")
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = (
            "You are given an image of a serial number label.\n"
            f"The OCR system extracted: `{ocr_serial}`\n"
            f"The GPT model extracted: `{gpt_serial}`\n\n"
            "Please determine the correct serial number based on the image and candidates. "
            "If neither is correct, reply with 'None'. Return only the final serial number or 'None'."
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
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            },
                        ],
                    }
                ],
                max_tokens=50,
            )
            answer = response.choices[0].message.content.strip()
            print(f"🧠 GPT verification result: {answer}")
            if answer.lower() == "none":
                return None
            return answer
        except Exception as e:
            print("❌ Error during GPT verification:", e)
            return None
