import streamlit as st
from meta_agent import MetaAgent
from damage_detection_agent import DamageDetectionAgent
from serial_number_agent import SerialNumberAgent
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from PIL import Image



st.title("Aircraft Inspection Assistant")

meta_agent = MetaAgent()
task_data = meta_agent.get_tasks_and_agents()

label_to_key = {v["label"]: k for k, v in task_data.items()}
selected_label = st.selectbox("Select inspection task", list(label_to_key.keys()))
task_type = label_to_key[selected_label]
st.write(f"Selected Task: {task_data[task_type]['label']}")
selected_agents = task_data[task_type]['agents']

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
                    "turns:fr-turn2.xirsys.com:5349?transport=tcp"
                ],
                "username": "irdXKd_EfhumLMw7XwVHh24u3OGGPh-ydOjPzM4C768Q-AShT6WNPCj0XmtXCeR7AAAAAGiRo79tb2FnaQ==",
                "credential": "ebb60e18-71c4-11f0-9bb8-fa2d218ee094"
            }
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

# Main app logic per agent
for agent in selected_agents:
    if agent == "SerialNumberAgent":
        sn_agent = SerialNumberAgent()
        st.subheader("Serial Number Agent")

        input_mode = st.radio("Choose Input Method", ["📷 Live Camera", "📁 Upload Image"])

        if input_mode == "📷 Live Camera":
            ctx = webrtc_streamer(
                key="serial-number",
                video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_config,
                async_processing=True,
                media_stream_constraints={
                    "video": True,
                    "audio": False  # ✅ This disables audio correctly
                }
            )

            if ctx.video_processor:
                scan_button_col1, scan_button_col2, scan_button_col3 = st.columns([3, 1, 3])
                with scan_button_col2:
                    scan_clicked = st.button("📸 Scan")

                if scan_clicked:
                    frame = ctx.video_processor.frame
                    if frame is not None:
                        pil_img = Image.fromarray(frame[..., ::-1])  # BGR → RGB
                        st.image(pil_img, caption="📷 Captured Frame", use_container_width=True)

                        with st.spinner("🔍 Analyzing image..."):
                            serial_number, conf = sn_agent.scan(pil_img)
                            if serial_number:
                                st.markdown("### 🧠 Detected Serial Number:")
                                st.success(f"`{serial_number}` (Confidence: {conf:.2f})")
                            else:
                                st.warning("No serial number detected.")
                    else:
                        st.warning("No frame captured. Please try again.")

        else:  # 📁 Upload Image
            uploaded_img = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_img:
                pil_img = Image.open(uploaded_img)
                st.image(pil_img, caption="🖼️ Uploaded Image", use_container_width=True)

                with st.spinner("🔍 Analyzing image..."):
                    serial_number, conf = sn_agent.scan(pil_img)
                    if serial_number:
                        st.markdown("### 🧠 Detected Serial Number:")
                        st.success(f"`{serial_number}` (Confidence: {conf:.2f})")
                    else:
                        st.warning("No serial number detected.")

