
import streamlit as st
from meta_agent import MetaAgent
from damage_detection_agent import DamageDetectionAgent
from serial_number_agent import SerialNumberAgent
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image

st.title("Aircraft Inspection Assistant")

meta_agent = MetaAgent()
task_data = meta_agent.get_tasks_and_agents()

label_to_key = {v["label"]: k for k, v in task_data.items()}
selected_label = st.selectbox("Select inspection task", list(label_to_key.keys()))
task_type = label_to_key[selected_label]
st.write(f"Selected Task: {task_data[task_type]['label']}")
selected_agents = task_data[task_type]['agents']

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

        ctx = webrtc_streamer(key="serial-number", video_processor_factory=VideoProcessor, async_processing=True)

        if ctx.video_processor:
            scan_button_col1, scan_button_col2, scan_button_col3 = st.columns([3, 1, 3])
            with scan_button_col2:
                scan_clicked = st.button("📸 Scan")

            if scan_clicked:
                frame = ctx.video_processor.frame
                if frame is not None:
                    content_col1, content_col2, content_col3 = st.columns([1, 2, 1])
                    with content_col2:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        st.image(pil_img, caption="📷 Captured Frame", use_container_width=True)
                        
                        with st.spinner("🔍 Analyzing image..."):
                            serial_number = sn_agent.scan(pil_img)
                            st.markdown("### 🧠 Detected Serial Number:")
                            st.write(f"`{serial_number}`")
                        
                else:
                    st.warning("No frame captured. Please try again.")

    elif agent == "DamageDetectionAgent":
        st.subheader("Damage Detection Agent")
        dd_agent = DamageDetectionAgent()
        st.info(dd_agent.get_status())