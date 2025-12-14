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
    """First-write-wins timestamp stamp into st.session_state.current_case[field]."""
    try:
        case = st.session_state.get("current_case")
        if case is None:
            return
        if case.get(field) is None:
            case[field] = _now_vienna_iso()
    except Exception:
        pass


# ===================== Agent =====================
class SerialNumberAgent:
    def __init__(
        self,
        api_url: str = "http://168.119.242.186:8500/scan_serial",
        ocr_early_accept_threshold: float = 0.95,
    ):
        self.api_url = api_url
        self.ocr_early_accept_threshold = float(ocr_early_accept_threshold)

    def scan(self, pil_img: Image.Image):
        """
        Pipeline (timestamps):
          - ts_ocr_result: OCR server response received
          - ts_gpt_result: GPT extraction response received (only if GPT is called)
          - ts_gpt_verification: GPT verification response received (only if GPT is called)
        """
        print("🔍 Starting serial number scan...")

        # 1) OCR
        ocr_serial, ocr_conf = self._try_ocr_api(pil_img)
        if ocr_conf is None:
            print("🚫 OCR server unavailable. Aborting serial number scan.")
            return None, 0.0

        print(f"📄 OCR result: {ocr_serial} (Confidence: {ocr_conf:.2f})")

        # ✅ Early accept: high confidence OCR -> skip GPT completely
        if ocr_serial and ocr_conf >= self.ocr_early_accept_threshold:
            print(f"✅ Early accept OCR (conf >= {self.ocr_early_accept_threshold:.2f}). Skipping GPT.")
            # Optional: stamp a "not executed" marker so your CSV stays easy to read
            # (leave them None if you prefer)
            # _stamp_case("ts_gpt_result")         # do NOT stamp
            # _stamp_case("ts_gpt_verification")  # do NOT stamp
            return ocr_serial, float(ocr_conf)

        # 2) GPT extraction
        gpt_serial = self._gpt_extract_serial(pil_img)
        print(f"🤖 GPT result: {gpt_serial}")

        # 3) GPT verification
        verified_serial = self._gpt_verify_serial(pil_img, ocr_serial, gpt_serial)
        print(f"🧪 GPT verification: {verified_serial}")

        if verified_serial:
            print(f"✅ Verified serial number: {verified_serial}")
            return verified_serial, float(ocr_conf)

        print("⚠️ No reliable serial number detected.")
        return None, 0.0

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

        prompt = (
            "Extract the serial number from this image. "
            "It may be labeled as SER', 'SERNO', 'SER NO', 'SERIAL', 'S/N', 'ESN', "
            "'Serial No', 'No Serie', etc. "
            "Ignore values labeled MODEL, TYPE, PNR, CERT, DATE, EXP, or MFR. "
            "Return only the serial number value."
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
            "Determine the correct serial number based on the image and the two candidates. "
            "If neither is correct, reply with 'None'. "
            "Return only the final serial number or 'None'."
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

            if answer.lower() == "none":
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
