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
from serial_number_knowledge_agent import SerialNumberKnowledgeAgent
from manual_serial_entry_agent import ManualSerialEntryAgent
from damage_detection_agent import DamageDetectionAgent

# CSV persistence helpers
from persistence import save_image, append_result, read_results

# ===================== Page setup =====================
st.set_page_config(page_title="Aircraft Inspection Assistant", layout="wide")
st.title("Aircraft Inspection Assistant")

# Optional: force video element to not grow too large (may affect other videos on page)
# st.markdown(
#     """
#     <style>
#       video { max-width: 520px !important; height: auto !important; }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.markdown("""
<style>

/* === MOBILE ONLY === */
@media (max-width: 768px) {

  /* Streamlit-webrtc container */
  div[data-testid="stVideo"] {
      max-width: 320px !important;
      margin-left: auto;
      margin-right: auto;
  }

  /* WebRTC internal video wrapper */
  div[data-testid="stVideo"] video {
      max-width: 320px !important;
      width: 320px !important;
      height: auto !important;
  }

  /* Captured image preview */
  img {
      max-width: 240px !important;
      height: auto !important;
  }
}

</style>
""", unsafe_allow_html=True)


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
    if not force and not autosave:
        return
    exp_id = st.session_state.get("experiment_id", "")
    img_path = save_image(pil_img) if pil_img is not None else None
    append_result({
        "experiment_id": exp_id,
        "task": task_key,
        "agent": agent_name,
        "input_type": input_type,
        "serial_number": serial_number,
        "confidence": float(conf) if conf is not None else None,
        "image_path": img_path,
        "notes": (notes.strip() if notes else None) or source_note,
    })
    st.success("✅ Result saved")

# ===================== Serial Number Agents =====================
def serial_number_interface(sn_agent, agent_name):
    st.subheader(f"{agent_name.replace('Agent','')} Inspection")
    mode = st.radio("Input", ["📷 Live Camera", "📁 Upload Image"], key=f"{agent_name}_mode")

    # --- State init ---
    for key in ["sn_detected_serial", "sn_conf", "sn_image", "sn_edit_mode", "sn_edit_value", "sn_last_upload_hash"]:
        st.session_state.setdefault(key, None if "edit_mode" not in key else False)

    def _sn_set_result(pil_img, serial_number, conf):
        st.session_state.sn_image = pil_img
        st.session_state.sn_detected_serial = serial_number
        st.session_state.sn_conf = conf
        st.session_state.sn_edit_value = serial_number or ""
        st.session_state.sn_edit_mode = False

    def _sn_save(serial_value, edited=False):
        source_note = "scanner-edited" if edited else "scanner-auto"
        _maybe_save(
            pil_img=st.session_state.sn_image,
            serial_number=serial_value,
            conf=st.session_state.sn_conf,
            input_type=("camera" if mode == "📷 Live Camera" else "upload"),
            agent_name=agent_name,
            task_key=task_type,
            force=True,
            source_note=source_note,
        )

    # --- Input: Camera or Upload ---
    if mode == "📷 Live Camera":
        # Layout: make the camera column narrower so the video doesn't fill the whole page
        col_cam, col_side = st.columns([1, 2], vertical_alignment="top")

        with col_cam:
            st.caption("Camera preview")
            ctx = webrtc_streamer(
                key=f"{agent_name}-cam",
                video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_config,
                async_processing=True,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                        "frameRate": {"ideal": 10},
                    },
                    "audio": False,
                },
            )

        with col_side:
            st.caption("Scan / Output")
            scan_clicked = st.button("📸 Scan", use_container_width=False)

            if ctx.video_processor and scan_clicked:
                frame = ctx.video_processor.frame
                if frame is not None:
                    pil_img = Image.fromarray(frame[..., ::-1])  # BGR -> RGB
                    with st.spinner("🔍 Analyzing image..."):
                        serial_number, conf = sn_agent.scan(pil_img)
                    if serial_number:
                        _sn_set_result(pil_img, serial_number, conf)
                        if conf == 1.0:
                            _sn_save(serial_number)
                            st.success("✅ Auto-saved (found in KnowledgeAgent).")
                        else:
                            st.success("Detected a serial number.")
                    else:
                        st.warning("No serial number detected.")
                else:
                    st.warning("No frame captured.")

            # Show last captured image/result on the side (optional but handy)
            if st.session_state.sn_image is not None:
                st.image(
                    st.session_state.sn_image,
                    caption="Last captured frame",
                    width=240   # 👈 adjust: 200–320 works well
                )

    else:
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_img:
            raw = uploaded_img.getvalue()
            current_hash = hashlib.md5(raw).hexdigest()
            pil_img = Image.open(io.BytesIO(raw))
            st.image(pil_img, caption="🖼️ Uploaded Image", use_container_width=True)
            if st.session_state.sn_last_upload_hash != current_hash:
                with st.spinner("🔍 Analyzing image..."):
                    serial_number, conf = sn_agent.scan(pil_img)
                if serial_number:
                    _sn_set_result(pil_img, serial_number, conf)
                    if conf == 1.0:
                        _sn_save(serial_number)
                        st.success("✅ Auto-saved (found in KnowledgeAgent).")
                    else:
                        st.success("Detected a serial number.")
                else:
                    st.warning("No serial number detected.")
                st.session_state.sn_last_upload_hash = current_hash

    # --- Review & Save ---
    st.divider()
    st.subheader("Review & Save")
    detected = st.session_state.sn_detected_serial
    conf = st.session_state.sn_conf

    if detected and conf != 1.0:
        st.markdown("**Detected Serial Number**")
        st.code(detected)
        if conf is not None:
            st.caption(f"Confidence: {conf:.2f}")
        col_a, col_b = st.columns([1, 1])
        if col_a.button("✅ Accept & Save"):
            _sn_save(detected, edited=False)
        if col_b.button("✏️ Edit"):
            st.session_state.sn_edit_mode = not st.session_state.sn_edit_mode
        if st.session_state.sn_edit_mode:
            new_val = st.text_input("Edit serial number", value=st.session_state.sn_edit_value)
            if st.button("💾 Save Edited Serial"):
                cleaned = new_val.strip()
                if cleaned:
                    _sn_save(cleaned, edited=True)
                    st.success(f"Saved edited serial number: `{cleaned}`")
                else:
                    st.warning("Please enter a valid serial number.")
    elif not detected:
        st.info("Capture or upload an image to detect the serial number.")

# ===================== Agent Selection =====================
if selected_agent == "SerialNumberAgent":
    serial_number_interface(SerialNumberAgent(), "SerialNumberAgent")

elif selected_agent == "SerialNumberKnowledgeAgent":
    serial_number_interface(SerialNumberKnowledgeAgent(), "SerialNumberKnowledgeAgent")

elif selected_agent == "ManualSerialEntryAgent":
    st.subheader("Manual Serial Entry")
    manual_agent = ManualSerialEntryAgent()
    manual_sn = st.text_input("Serial Number", placeholder="e.g., ABC1234567")
    if st.button("💾 Save Manual Entry"):
        if manual_agent.validate(manual_sn):
            _maybe_save(
                pil_img=None,
                serial_number=manual_sn.strip(),
                conf=None,
                input_type="manual",
                agent_name="ManualSerialEntryAgent",
                task_key=task_type,
                force=True
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
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="experiments.csv",
        mime="text/csv",
    )
else:
    st.info("No results saved yet.")
