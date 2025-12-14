import os
import io
import base64
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from PIL import Image
import openai
import streamlit as st
from dotenv import load_dotenv

from knowledge_agent import KnowledgeAgent


# ===================== OpenAI key =====================
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except st.runtime.secrets.StreamlitSecretNotFoundError:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


# ===================== Timestamp helpers =====================
VIENNA = ZoneInfo("Europe/Vienna")


def _now_vienna_iso() -> str:
    return datetime.now(VIENNA).isoformat(timespec="milliseconds")


def _stamp_case(field: str) -> None:
    """
    Stamp a timestamp into the currently active test case (st.session_state.current_case).
    First-write-wins: never overwrite a value that is already present.
    """
    try:
        case = st.session_state.get("current_case")
        if case is None:
            return
        if case.get(field) is None:
            case[field] = _now_vienna_iso()
    except Exception:
        # keep agent usable outside Streamlit
        pass


# ===================== Agent =====================
class SerialNumberKnowledgeAgent:
    """
    Pipeline (timestamps):
      - ts_ocr_result: OCR server response received
      - ts_gpt_result: GPT extraction response received
      - ts_gpt_verification: GPT verification response received

    Behavior:
      1) OCR via API
      2) Knowledge list check (auto-accept if match)
      3) GPT extraction
      4) Knowledge list check (auto-accept if match)
      5) GPT verification (final arbitration)
      6) Knowledge list check (auto-accept if match)

    Return signature (for UI):
      (serial_number, ocr_confidence, is_known_good, source)

    Where:
      - is_known_good == True means: match in KnowledgeAgent list -> UI should auto-save
      - source in {"ocr", "gpt", "verify", "none"}
      - confidence is ALWAYS PaddleOCR confidence
    """

    def __init__(self, api_url: str = "http://168.119.242.186:8500/scan_serial"):
        self.api_url = api_url
        self.knowledge_agent = KnowledgeAgent()

    def scan(self, pil_img: Image.Image):
        print("🔍 Starting knowledge-based serial number scan...")

        important = set(self.knowledge_agent.get_important_serials())

        # 1) OCR
        ocr_serial, ocr_conf = self._try_ocr_api(pil_img)
        if ocr_conf is None:
            print("🚫 OCR server unavailable. Aborting scan.")
            return None, 0.0, False, "none"

        if ocr_serial:
            print(f"📄 OCR result: {ocr_serial} (Confidence: {ocr_conf:.2f})")

            # 2) Knowledge check after OCR
            if ocr_serial in important:
                print("✅ OCR matches Knowledge list. Auto-accepting.")
                return ocr_serial, float(ocr_conf), True, "ocr"

        # 3) GPT extraction
        gpt_serial = self._gpt_extract_serial(pil_img)
        if gpt_serial:
            print(f"🤖 GPT result: {gpt_serial}")

            # 4) Knowledge check after GPT extraction
            if gpt_serial in important:
                print("✅ GPT extraction matches Knowledge list. Auto-accepting.")
                return gpt_serial, float(ocr_conf), True, "gpt"

        # 5) GPT verification
        verified = self._gpt_verify_serial(pil_img, ocr_serial, gpt_serial)
        if verified:
            print(f"🧪 Verified serial number: {verified}")

            # 6) Knowledge check after verification
            if verified in important:
                print("✅ GPT verification matches Knowledge list. Auto-accepting.")
                return verified, float(ocr_conf), True, "verify"

            # Not in list -> user must accept/edit
            return verified, float(ocr_conf), False, "verify"

        print("⚠️ No reliable serial number detected.")
        return None, float(ocr_conf), False, "none"

    # ----------------- internal helpers -----------------

    def _try_ocr_api(self, pil_img: Image.Image):
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        buffered.seek(0)
        files = {"file": ("image.jpg", buffered, "image/jpeg")}

        try:
            response = requests.post(self.api_url, files=files, timeout=10)

            if response.status_code != 200:
                print(f"❌ OCR API error: {response.status_code} - {response.text}")
                return None, None

            data = response.json()
            serial_number = data.get("serial_number")
            confidence = data.get("confidence", 0.0)

            # ✅ OCR result returned from server
            _stamp_case("ts_ocr_result")

            return serial_number, confidence

        except Exception as e:
            print("❌ OCR API not reachable:", e)
            return None, None

    def _gpt_extract_serial(self, pil_img: Image.Image):
        img_base64 = self._pil_to_base64_png(pil_img)

        # ✅ Improved prompt (label not part of the serial)
        prompt = (
            "Extract the serial number from this image.\n\n"
            "IMPORTANT:\n"
            "- Return ONLY the serial number value (alphanumeric).\n"
            "- DO NOT include labels like SER, SERIAL, SER NO, SERNO, S/N, ESN, SN, NO., etc.\n"
            "- Ignore values labeled MODEL, TYPE, P/N, PN, PART NO, PNR, CERT, DATE, EXP, or MFR.\n"
            "- If no clear serial number is visible, return 'None'."
        )

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    ],
                }],
                max_tokens=50,
            )
            out = response.choices[0].message.content.strip()

            # ✅ GPT extraction returned
            _stamp_case("ts_gpt_result")

            if out and out.lower() == "none":
                return None
            return out

        except Exception as e:
            print("❌ Error calling GPT for extraction:", e)
            return None

    def _gpt_verify_serial(self, pil_img: Image.Image, ocr_serial: str, gpt_serial: str):
        img_base64 = self._pil_to_base64_png(pil_img)

        prompt = (
            "You are given an image of a serial number label.\n"
            f"The OCR system extracted: `{ocr_serial}`\n"
            f"The GPT model extracted: `{gpt_serial}`\n\n"
            "Choose the correct serial number based on the image.\n"
            "Return ONLY the final serial number value (no labels like SER/SN/S/N), or 'None'."
        )

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    ],
                }],
                max_tokens=50,
            )
            answer = response.choices[0].message.content.strip()

            # ✅ GPT verification returned
            _stamp_case("ts_gpt_verification")

            if answer and answer.lower() == "none":
                return None
            return answer

        except Exception as e:
            print("❌ Error calling GPT for verification:", e)
            return None

    @staticmethod
    def _pil_to_base64_png(pil_img: Image.Image) -> str:
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
