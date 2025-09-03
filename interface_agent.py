# interface_agent.py
import uuid
import io
import hashlib
import pandas as pd
import streamlit as st
from PIL import Image
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- your own modules ---
from meta_agent import MetaAgent
from serial_number_agent import SerialNumberAgent
from manual_serial_entry_agent import ManualSerialEntryAgent
from damage_detection_agent import DamageDetectionAgent

# CSV persistence helpers
from persistence import save_image, append_result, read_results


# ===================== Page setup =====================
st.set_page_config(page_title="Aircraft Inspection Assistant", layout="wide")
st.title("Aircraft Inspection Assistant")

# --- Stable Experiment ID (once per session) ---
if "experiment_id" not in st.session_state:
    st.session_state.experiment_id = str(uuid.uuid4())[:8]

# Sidebar: experiment/session controls
with st.sidebar:
    st.header("Experiment")
    st.text_input("Experiment ID", key="experiment_id")  # stays stable across reruns
    autosave = st.toggle("Auto-save results", value=True)
    notes = st.text_area("Notes (optional)")
    if st.button("🔁 New random ID"):
        st.session_state.experiment_id = str(uuid.uuid4())[:8]


# ===================== Meta agent / tasks =====================
meta_agent = MetaAgent()
task_data = meta_agent.get_tasks_and_agents()

# Dropdown of tasks by label
label_to_key = {v["label"]: k for k, v in task_data.items()}
selected_label = st.selectbox("Select inspection task", list(label_to_key.keys()))
task_type = label_to_key[selected_label]
st.write(f"Selected Task: {task_data[task_type]['label']}")

# For this setup, each task has exactly one agent
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
                "username": "irdXKd_EfhumLMw7XwVHh24u3OGGPh-ydOjPzM4C768Q-AShT6WNPCj0XmtXCeR7AAAAAGiRo79tb2FnaQ==",
                "credential": "ebb60e18-71c4-11f0-9bb8-fa2d218ee094",
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


# ===================== Helper: save one result row =====================
def _maybe_save(*, pil_img, serial_number, conf, input_type, agent_name, task_key, force=False, source_note=None):
    if not force and not autosave:
        return

    exp_id = st.session_state.get("experiment_id", "")
    img_path = save_image(pil_img) if pil_img is not None else None

    append_result(
        {
            "experiment_id": exp_id,
            "task": task_key,
            "agent": agent_name,
            "input_type": input_type,   # "camera" | "upload" | "manual"
            "serial_number": serial_number,
            "confidence": float(conf) if conf is not None else None,
            "image_path": img_path,
            "notes": (notes.strip() if notes else None) or source_note,  # ✅ auto remark
        }
    )
    st.success("✅ Result saved")



# ===================== Render the selected task/agent =====================
if selected_agent == "SerialNumberAgent":
    # --- Serial Number Inspection with review → accept/edit → save ---
    st.subheader("Serial Number Inspection")

    # Sub-choice: live camera or upload
    mode = st.radio("Input", ["📷 Live Camera", "📁 Upload Image"], key="sn_mode")
    sn_agent = SerialNumberAgent()

    # ---------- session state defaults ----------
    def _sn_init_state():
        st.session_state.setdefault("sn_detected_serial", None)
        st.session_state.setdefault("sn_conf", None)
        st.session_state.setdefault("sn_image", None)
        st.session_state.setdefault("sn_edit_mode", False)
        st.session_state.setdefault("sn_edit_value", "")
        st.session_state.setdefault("sn_last_upload_hash", None)

    _sn_init_state()

    def _sn_set_result(pil_img, serial_number, conf):
        st.session_state.sn_image = pil_img
        st.session_state.sn_detected_serial = serial_number
        st.session_state.sn_conf = conf
        st.session_state.sn_edit_mode = False
        st.session_state.sn_edit_value = serial_number or ""

    def _sn_save(serial_value, edited=False):
        source_note = "scanner-edited" if edited else "scanner-auto"
        _maybe_save(
            pil_img=st.session_state.sn_image,
            serial_number=serial_value,
            conf=st.session_state.sn_conf,
            input_type=("camera" if mode == "📷 Live Camera" else "upload"),
            agent_name="SerialNumberAgent",
            task_key=task_type,
            force=True,
            source_note=source_note,   # ✅ mark in CSV
    )

    # ================= Input panes =================
    if mode == "📷 Live Camera":
        ctx = webrtc_streamer(
            key="serial-number",
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.video_processor:
            c1, c2, c3 = st.columns([3, 1, 3])
            with c2:
                scan_clicked = st.button("📸 Scan", key="sn_scan_camera")
            if scan_clicked:
                frame = ctx.video_processor.frame
                if frame is not None:
                    pil_img = Image.fromarray(frame[..., ::-1])
                    with st.spinner("🔍 Analyzing image..."):
                        serial_number, conf = sn_agent.scan(pil_img)
                    if serial_number:
                        st.success("Detected a serial number.")
                        _sn_set_result(pil_img, serial_number, conf)
                    else:
                        st.warning("No serial number detected.")
                        _sn_set_result(pil_img, None, 0.0)
                else:
                    st.warning("No frame captured. Please try again.")

    else:  # 📁 Upload Image
        uploaded_img = st.file_uploader(
            "Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"], key="sn_upload"
        )
        if uploaded_img:
            raw = uploaded_img.getvalue()
            current_hash = hashlib.md5(raw).hexdigest()
            pil_img = Image.open(io.BytesIO(raw))
            st.image(pil_img, caption="🖼️ Uploaded Image", use_container_width=True)

            if st.session_state.sn_last_upload_hash != current_hash:
                with st.spinner("🔍 Analyzing image..."):
                    serial_number, conf = sn_agent.scan(pil_img)
                if serial_number:
                    st.success("Detected a serial number.")
                    _sn_set_result(pil_img, serial_number, conf)
                else:
                    st.warning("No serial number detected.")
                    _sn_set_result(pil_img, None, 0.0)
                st.session_state.sn_last_upload_hash = current_hash

    # ================= Review & Save panel =================
    st.divider()
    st.subheader("Review & Save")

    detected = st.session_state.sn_detected_serial
    conf = st.session_state.sn_conf

    if detected:
        st.markdown("**Detected Serial Number**")
        st.code(detected)
        if conf is not None:
            st.caption(f"Confidence: {conf:.2f}")

        col_a, col_b, col_c = st.columns([1, 1, 4])
        with col_a:
            accept_clicked = st.button("✅ Accept & Save", key="sn_accept_save", use_container_width=True)
        with col_b:
            edit_clicked = st.button(
                ("✏️ Edit" if not st.session_state.sn_edit_mode else "❌ Cancel Edit"),
                key="sn_toggle_edit",
                use_container_width=True,
            )

        if accept_clicked and not st.session_state.sn_edit_mode:
            _sn_save(detected, edited=False)

        if edit_clicked:
            st.session_state.sn_edit_mode = not st.session_state.sn_edit_mode
            if st.session_state.sn_edit_mode:
                st.session_state.sn_edit_value = detected

        if st.session_state.sn_edit_mode:
            new_val = st.text_input(
                "Edit serial number",
                value=st.session_state.sn_edit_value,
                key="sn_edit_input",
            )
            col_s, _ = st.columns([1, 5])
            with col_s:
                save_edit_clicked = st.button("💾 Save Edited Serial", key="sn_save_edit", use_container_width=True)
            if save_edit_clicked:
                cleaned = (new_val or "").strip()
                if cleaned:
                    _sn_save(cleaned, edited=True)
                    st.session_state.sn_detected_serial = cleaned
                    st.session_state.sn_edit_mode = False
                    st.success(f"Saved edited serial number: `{cleaned}`")
                else:
                    st.warning("Please enter a valid serial number before saving.")
    else:
        st.info("Capture a frame or upload an image to detect the serial number. Nothing will be saved until you accept or edit & save.")

elif selected_agent == "ManualSerialEntryAgent":
    st.subheader("Manual Serial Entry")
    manual_agent = ManualSerialEntryAgent()
    st.info("Enter a serial number manually. This will be saved with input_type=manual (no image).")

    manual_sn = st.text_input("Serial Number", placeholder="e.g., ABC1234567", key="manual_sn")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        save_clicked = st.button("💾 Save Manual Entry", key="manual_save")

    if save_clicked:
        if manual_agent.validate(manual_sn):
            cleaned = manual_sn.strip()
            _maybe_save(
                pil_img=None,
                serial_number=cleaned,
                conf=None,
                input_type="manual",
                agent_name="ManualSerialEntryAgent",
                task_key=task_type,
                force=True,
            )
            st.success(f"Saved manual serial number: `{cleaned}`")
        else:
            st.warning("Please enter a valid serial number.")

elif selected_agent == "DamageDetectionAgent":
    st.subheader("Damage Detection")
    dd_agent = DamageDetectionAgent()
    st.info(dd_agent.get_status())


# ===================== Results viewer & export =====================
st.divider()
st.subheader("Saved Results")

rows = read_results()
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="experiments.csv",
        mime="text/csv",
    )
else:
    st.info("No results saved yet.")
