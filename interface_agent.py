import uuid
import io
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from PIL import Image
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- your own modules ---
from meta_agent import MetaAgent
from serial_number_agent import SerialNumberAgent
from serial_number_knowledge_agent import SerialNumberKnowledgeAgent
from manual_serial_entry_agent import ManualSerialEntryAgent
from damage_detection_agent import DamageDetectionAgent
from scanner_agent import ScannerAgent

# CSV persistence helpers
from persistence import save_image, append_result, read_results


# ===================== Page setup =====================
st.set_page_config(page_title="Aircraft Inspection Assistant", layout="wide")
st.title("Aircraft Inspection Assistant")

st.markdown(
    """
<style>
@media (max-width: 768px) {
  div[data-testid="stVideo"] { max-width: 320px !important; margin-left: auto; margin-right: auto; }
  div[data-testid="stVideo"] video { max-width: 320px !important; width: 320px !important; height: auto !important; }
  img { max-width: 240px !important; height: auto !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ===================== Case timeline (stored into Saved Results) =====================
VIENNA = ZoneInfo("Europe/Vienna")


def now_vienna_iso() -> str:
    return datetime.now(VIENNA).isoformat(timespec="milliseconds")


def init_case_state():
    st.session_state.setdefault("current_case", None)
    st.session_state.setdefault("webrtc_was_playing", False)


def start_new_case(trigger: str, *, task_key: str, agent_name: str, input_type: str, **meta):
    """
    Start a new test-case timeline.
    One case -> one row in Saved Results.
    """
    st.session_state.current_case = {
        "case_id": str(uuid.uuid4())[:8],
        "experiment_id": st.session_state.get("experiment_id", ""),
        "task": task_key,
        "agent": agent_name,
        "input_type": input_type,
        "trigger": trigger,
        "meta": json.dumps(meta, ensure_ascii=False) if meta else None,
        # timeline columns (all go into Saved Results)
        "ts_camera_start": None,
        "ts_scan_pressed": None,
        "ts_ocr_result": None,           # stamped inside agent (OCR API returns)
        "ts_gpt_result": None,           # stamped inside agent (GPT extract returns)
        "ts_gpt_verification": None,     # stamped inside agent (GPT verify returns)
        "ts_accept_save_pressed": None,
        "ts_edit_pressed": None,
        "ts_save_edited_pressed": None,
        "ts_result_saved": None,
    }


def ensure_case(*, task_key: str, agent_name: str, input_type: str, trigger_if_new: str = "auto"):
    init_case_state()
    if st.session_state.current_case is None:
        start_new_case(trigger_if_new, task_key=task_key, agent_name=agent_name, input_type=input_type)


def stamp(field: str, *, task_key: str, agent_name: str, input_type: str):
    """
    UI-side stamps (camera start / scan pressed / accept/save/edit/save edited / result saved).
    First-write-wins to avoid overwriting agent stamps.
    """
    ensure_case(task_key=task_key, agent_name=agent_name, input_type=input_type)
    if st.session_state.current_case.get(field) is None:
        st.session_state.current_case[field] = now_vienna_iso()


init_case_state()

# --- Stable Experiment ID ---
if "experiment_id" not in st.session_state:
    st.session_state.experiment_id = str(uuid.uuid4())[:8]

# Sidebar controls
with st.sidebar:
    st.header("Experiment")
    st.text_input("Experiment ID", key="experiment_id")
    autosave = st.toggle("Auto-save results", value=True)
    notes = st.text_area("Notes (optional)")
    if st.button("🔁 New random ID"):
        st.session_state.experiment_id = str(uuid.uuid4())[:8]


# ===================== Meta agent / tasks =====================
meta_agent = MetaAgent()
task_data = meta_agent.get_tasks_and_agents()
label_to_key = {v["label"]: k for k, v in task_data.items()}
selected_label = st.selectbox("Select inspection task", list(label_to_key.keys()))
task_type = label_to_key[selected_label]
st.write(f"Selected Task: {task_data[task_type]['label']}")
selected_agent = task_data[task_type]["agents"][0]


# ===================== WebRTC config =====================
rtc_config = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:fr-turn2.xirsys.com"]},
            {
                "urls": [
                    "turn:fr-turn2.xirsys.com:80?transport=udp",
                    "turn:fr-turn2.xirsys.com:3478?transport=udp",
                    "turn:fr-turn2.xirsys.com:80?transport=tcp",
                    "turn:fr-turn2.xirsys.com:3478?transport=tcp",
                    "turns:fr-turn2.xirsys.com:443?transport=tcp",
                    "turns:fr-turn2.xirsys.com:5349?transport=tcp",
                ],
                "username": "vqm0jqS7--XPotXOOQaxUaweKI4ox5JEALGKbWyorSGUZIaGZ5Ffj5fJIs33aai6AAAAAGi3_69lMTE3Nzk2MTQ=",
                "credential": "0f9475b0-88a2-11f0-9318-b26d1547bb7d",
            },
        ]
    }
)


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ===================== Helper =====================
def _maybe_save(*, pil_img, serial_number, conf, input_type, agent_name, task_key, force=False, source_note=None):
    """
    Save ONE row to Saved Results.
    Also merges timeline stamps from the current test case into that row.
    """
    if not force and not autosave:
        return

    stamp("ts_result_saved", task_key=task_key, agent_name=agent_name, input_type=input_type)

    exp_id = st.session_state.get("experiment_id", "")
    img_path = save_image(pil_img) if pil_img is not None else None

    base_row = {
        "experiment_id": exp_id,
        "case_id": (st.session_state.current_case or {}).get("case_id"),
        "task": task_key,
        "agent": agent_name,
        "input_type": input_type,
        "serial_number": serial_number,
        "confidence": float(conf) if conf is not None else None,
        "image_path": img_path,
        "notes": (notes.strip() if notes else None) or source_note,
    }

    timeline_cols = {}
    if st.session_state.current_case:
        timeline_cols = {k: v for k, v in st.session_state.current_case.items() if k.startswith("ts_")}

    append_result({**base_row, **timeline_cols})
    st.success("✅ Result saved")

    st.session_state.current_case = None


# ===================== Serial Number UI =====================
def _unpack_agent_result(result):
    """
    Supports:
      - (serial, conf)
      - (serial, conf, is_known_good, source) [Knowledge agent]
    Returns:
      serial, conf, is_known_good, source
    """
    serial = None
    conf = 0.0
    is_known_good = False
    source = None

    if isinstance(result, tuple) and len(result) == 4:
        serial, conf, is_known_good, source = result
    else:
        serial, conf = result

    return serial, conf, bool(is_known_good), source


def serial_number_interface(sn_agent, agent_name: str):
    """
    - SerialNumberAgent: manual Accept/Edit
    - SerialNumberKnowledgeAgent: auto-save ONLY if is_known_good True
    - ScannerAgent: auto-save always (sn_agent.is_auto_save True)
    """
    st.subheader(f"{agent_name.replace('Agent','')} Inspection")

    auto_save_mode = bool(getattr(sn_agent, "is_auto_save", False))
    mode = st.radio("Input", ["📷 Live Camera", "📁 Upload Image"], key=f"{agent_name}_mode")
    input_type = "camera" if mode == "📷 Live Camera" else "upload"

    # --- State init ---
    for key in ["sn_detected_serial", "sn_conf", "sn_image", "sn_edit_mode", "sn_edit_value", "sn_last_upload_hash"]:
        st.session_state.setdefault(key, None if "edit_mode" not in key else False)
    st.session_state.setdefault("sn_is_known_good", False)
    st.session_state.setdefault("sn_source", None)

    def _clear_sn_state():
        st.session_state.sn_detected_serial = None
        st.session_state.sn_conf = None
        st.session_state.sn_image = None
        st.session_state.sn_edit_mode = False
        st.session_state.sn_edit_value = ""
        st.session_state.sn_is_known_good = False
        st.session_state.sn_source = None

    def _sn_set_result(pil_img, serial_number, conf, *, is_known_good=False, source=None):
        st.session_state.sn_image = pil_img
        st.session_state.sn_detected_serial = serial_number
        st.session_state.sn_conf = conf
        st.session_state.sn_edit_value = serial_number or ""
        st.session_state.sn_edit_mode = False
        st.session_state.sn_is_known_good = bool(is_known_good)
        st.session_state.sn_source = source

    def _sn_save(serial_value, edited=False, note_override=None):
        source_note = note_override or ("scanner-edited" if edited else "scanner-auto")
        _maybe_save(
            pil_img=st.session_state.sn_image,
            serial_number=serial_value,
            conf=st.session_state.sn_conf,
            input_type=input_type,
            agent_name=agent_name,
            task_key=task_type,
            force=True,
            source_note=source_note,
        )

    # ===================== CAMERA =====================
    if mode == "📷 Live Camera":
        col_cam, col_side = st.columns([1, 2], vertical_alignment="top")

        with col_cam:
            st.caption("Camera preview")
            ctx = webrtc_streamer(
                key=f"{agent_name}-cam",
                video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_config,
                async_processing=True,
                media_stream_constraints={
                    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 10}},
                    "audio": False,
                },
            )

            is_playing = bool(getattr(ctx.state, "playing", False)) if ctx else False
            was_playing = st.session_state.get("webrtc_was_playing", False)

            if is_playing and not was_playing:
                start_new_case(trigger="camera_start", task_key=task_type, agent_name=agent_name, input_type=input_type)
                stamp("ts_camera_start", task_key=task_type, agent_name=agent_name, input_type=input_type)

            st.session_state.webrtc_was_playing = is_playing

        with col_side:
            st.caption("Scan / Output")
            scan_clicked = st.button("📸 Scan", use_container_width=False)

            if scan_clicked:
                ensure_case(task_key=task_type, agent_name=agent_name, input_type=input_type, trigger_if_new="scan_pressed")
                stamp("ts_scan_pressed", task_key=task_type, agent_name=agent_name, input_type=input_type)

            if ctx.video_processor and scan_clicked:
                frame = ctx.video_processor.frame
                if frame is None:
                    st.warning("No frame captured.")
                else:
                    pil_img = Image.fromarray(frame[..., ::-1])  # BGR -> RGB

                    with st.spinner("🔍 Analyzing image..."):
                        result = sn_agent.scan(pil_img)

                    serial_number, conf, is_known_good, source = _unpack_agent_result(result)

                    if serial_number:
                        _sn_set_result(pil_img, serial_number, conf, is_known_good=is_known_good, source=source)

                        # Scanner mode: always auto-save
                        if auto_save_mode:
                            _sn_save(serial_number, edited=False, note_override="scanner-auto")
                            st.success("✅ Auto-saved (Scanner mode).")
                            _clear_sn_state()
                            st.stop()

                        # Knowledge mode: auto-save ONLY if knowledge match
                        if is_known_good:
                            _sn_save(serial_number, edited=False, note_override=f"knowledge-auto:{source}")
                            st.success(f"✅ Found in KnowledgeAgent ({source}) → saved automatically.")
                            _clear_sn_state()
                            st.stop()

                        st.success("Detected a serial number.")
                    else:
                        st.warning("No serial number detected.")

            if st.session_state.sn_image is not None:
                st.image(st.session_state.sn_image, caption="Last captured frame", width=240)

    # ===================== UPLOAD =====================
    else:
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_img:
            raw = uploaded_img.getvalue()
            current_hash = hashlib.md5(raw).hexdigest()
            pil_img = Image.open(io.BytesIO(raw))
            st.image(pil_img, caption="🖼️ Uploaded Image", use_container_width=True)

            if st.session_state.sn_last_upload_hash != current_hash:
                start_new_case(
                    trigger="upload_analyzed",
                    task_key=task_type,
                    agent_name=agent_name,
                    input_type=input_type,
                    bytes=len(raw),
                )

                with st.spinner("🔍 Analyzing image..."):
                    result = sn_agent.scan(pil_img)

                serial_number, conf, is_known_good, source = _unpack_agent_result(result)

                if serial_number:
                    _sn_set_result(pil_img, serial_number, conf, is_known_good=is_known_good, source=source)

                    if auto_save_mode:
                        _sn_save(serial_number, edited=False, note_override="scanner-auto")
                        st.success("✅ Auto-saved (Scanner mode).")
                        _clear_sn_state()
                        st.stop()

                    if is_known_good:
                        _sn_save(serial_number, edited=False, note_override=f"knowledge-auto:{source}")
                        st.success(f"✅ Found in KnowledgeAgent ({source}) → saved automatically.")
                        _clear_sn_state()
                        st.stop()

                    st.success("Detected a serial number.")
                else:
                    st.warning("No serial number detected.")

                st.session_state.sn_last_upload_hash = current_hash

    # ===================== REVIEW / SAVE =====================
    if auto_save_mode:
        st.divider()
        st.info("Scanner mode: results are saved automatically after each scan.")
        return

    st.divider()
    st.subheader("Review & Save")

    detected = st.session_state.sn_detected_serial
    conf = st.session_state.sn_conf

    if detected:
        st.markdown("**Detected Serial Number**")
        st.code(detected)

        if conf is not None:
            try:
                st.caption(f"Confidence (PaddleOCR): {float(conf):.3f}")
            except Exception:
                st.caption(f"Confidence (PaddleOCR): {conf}")

        col_a, col_b = st.columns([1, 1])

        if col_a.button("✅ Accept & Save"):
            stamp("ts_accept_save_pressed", task_key=task_type, agent_name=agent_name, input_type=input_type)
            _sn_save(detected, edited=False, note_override="manual-accept")

        if col_b.button("✏️ Edit"):
            stamp("ts_edit_pressed", task_key=task_type, agent_name=agent_name, input_type=input_type)
            st.session_state.sn_edit_mode = not st.session_state.sn_edit_mode

        if st.session_state.sn_edit_mode:
            new_val = st.text_input("Edit serial number", value=st.session_state.sn_edit_value)
            if st.button("💾 Save Edited Serial"):
                stamp("ts_save_edited_pressed", task_key=task_type, agent_name=agent_name, input_type=input_type)
                cleaned = new_val.strip()
                if cleaned:
                    _sn_save(cleaned, edited=True, note_override="manual-edited")
                    st.success(f"Saved edited serial number: `{cleaned}`")
                else:
                    st.warning("Please enter a valid serial number.")
    else:
        st.info("Capture or upload an image to detect the serial number.")


# ===================== Agent Selection =====================
if selected_agent == "SerialNumberAgent":
    serial_number_interface(SerialNumberAgent(), "SerialNumberAgent")

elif selected_agent == "SerialNumberKnowledgeAgent":
    serial_number_interface(SerialNumberKnowledgeAgent(), "SerialNumberKnowledgeAgent")

elif selected_agent == "ScannerAgent":
    serial_number_interface(ScannerAgent(), "ScannerAgent")

elif selected_agent == "ManualSerialEntryAgent":
    st.subheader("Manual Serial Entry")
    manual_agent = ManualSerialEntryAgent()
    manual_sn = st.text_input("Serial Number", placeholder="e.g., ABC1234567")

    if st.button("💾 Save Manual Entry"):
        start_new_case(trigger="manual_entry", task_key=task_type, agent_name="ManualSerialEntryAgent", input_type="manual")
        if manual_agent.validate(manual_sn):
            _maybe_save(
                pil_img=None,
                serial_number=manual_sn.strip(),
                conf=None,
                input_type="manual",
                agent_name="ManualSerialEntryAgent",
                task_key=task_type,
                force=True,
            )
            st.success(f"Saved manual serial number: `{manual_sn.strip()}`")
        else:
            st.warning("Please enter a valid serial number.")

elif selected_agent == "DamageDetectionAgent":
    st.subheader("Damage Detection")
    dd_agent = DamageDetectionAgent()
    st.info(dd_agent.get_status())


# ===================== Results Viewer =====================
st.divider()
st.subheader("Saved Results")

rows = read_results()
if rows:
    df = pd.DataFrame(rows)

    preferred = [
        "experiment_id",
        "case_id",
        "task",
        "agent",
        "input_type",
        "serial_number",
        "confidence",
        "ts_camera_start",
        "ts_scan_pressed",
        "ts_ocr_result",
        "ts_gpt_result",
        "ts_gpt_verification",
        "ts_accept_save_pressed",
        "ts_edit_pressed",
        "ts_save_edited_pressed",
        "ts_result_saved",
        "image_path",
        "notes",
        "timestamp_iso",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="experiments.csv",
        mime="text/csv",
    )
else:
    st.info("No results saved yet.")
