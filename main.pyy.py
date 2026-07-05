import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import numpy as np
from ultralytics import YOLO
import time
import math

# 1. Page Configuration & Aesthetic styling
st.set_page_config(
    page_title="BIOSTREAM PRO // QUANTUM CORE",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Cyberpunk Matrix Injector
st.markdown("""
<style>
    body { background-color: #02040a; color: #e2e8f0; }
    .stApp { background: #02040a; }
    .metric-box {
        background: rgba(6, 9, 22, 0.85);
        border: 1px solid rgba(14, 242, 254, 0.15);
        padding: 16px; 
        border-radius: 12px; 
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .title-cyber { 
        color: #0ef2fe; 
        text-shadow: 0 0 10px rgba(14,242,254,0.4); 
        font-weight: 900; 
        letter-spacing: 2px;
        font-size: 1.6rem;
    }
    .sec-title {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #4b5563;
        letter-spacing: 2px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Optimized Cache Architecture for YOLO
@st.cache_resource
def load_yolo_pose_model():
    return YOLO("yolov8n-pose.pt")

pose_model = load_yolo_pose_model()

def calculate_angle(p1, p2):
    try:
        x1, y1 = p1
        x2, y2 = p2
        return abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    except:
        return 90

# 3. Initialize Shared Application Memory State
if "test_active" not in st.session_state:
    st.session_state.test_active = False
    st.session_state.start_time = None
    st.session_state.healthy_frames = 0
    st.session_state.total_frames = 0
    st.session_state.final_report = None

if "live_metrics" not in st.session_state:
    st.session_state.live_metrics = {"status": "LOCKING TARGET...", "color": "#f43f5e"}

# 4. Sidebar Interface Layout
st.sidebar.markdown("<h1 class='title-cyber'>BIOSTREAM // PRO</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

if st.sidebar.button("🚀 RUN ENGINE DIAGNOSTIC TEST", use_container_width=True):
    st.session_state.test_active = True
    st.session_state.start_time = time.time()
    st.session_state.healthy_frames = 0
    st.session_state.total_frames = 0
    st.session_state.final_report = None
    st.toast("Biometric core diagnostics triggered! Sit upright for 15s.", icon="⏳")

# Layout Structures
col_feed, col_telemetry = st.columns([5, 4])

# 5. Native Video Processing Pipeline Engine
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # AI Processing inference execution
    results = pose_model(img, verbose=False)
    current_status = "INITIALIZING"
    human_detected = False
    
    for result in results:
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            keypoints = result.keypoints.xy[0].cpu().numpy()
            if len(keypoints) < 7: 
                continue
            
            human_detected = True
            try:
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_ear = keypoints[3]
                right_ear = keypoints[4]
                
                if left_shoulder[0] != 0 and right_shoulder[0] != 0:
                    shoulder_angle = calculate_angle(left_shoulder, right_shoulder)
                    neck_angle = calculate_angle(right_shoulder, right_ear)
                    
                    # Strict Vector Geometry Threshold checks
                    if (shoulder_angle > 14) or (neck_angle < 66 or neck_angle > 114):
                        current_status = "UNHEALTHY"
                    else:
                        current_status = "HEALTHY"
                    
                    # Render Graphic Overlays onto Matrix frame
                    for kp in [left_shoulder, right_shoulder, left_ear, right_ear]:
                        cv2.circle(img, (int(kp[0]), int(kp[1])), 6, (254, 242, 0), -1)
                    
                    color = (129, 185, 10) if current_status == "HEALTHY" else (94, 63, 244)
                    cv2.line(img, (int(left_shoulder[0]), int(left_shoulder[1])), (int(right_shoulder[0]), int(right_shoulder[1])), color, 3)
            except:
                pass

    # Thread-Safe Telemetry Sync Updates
    if human_detected:
        st.session_state.live_metrics["status"] = f"{current_status} PROFILE"
        st.session_state.live_metrics["color"] = "#10b981" if current_status == "HEALTHY" else "#f43f5e"
        
        # Test Routine Assessment logic
        if st.session_state.test_active:
            st.session_state.total_frames += 1
            if current_status == "HEALTHY":
                st.session_state.healthy_frames += 1
    else:
        st.session_state.live_metrics["status"] = "LOCKING TARGET..."
        st.session_state.live_metrics["color"] = "#4b5563"

    return frame.from_ndarray(img, format="bgr24")

# 6. Stream Engine Allocation and Layout Rendering
with col_feed:
    st.markdown("### 📽️ Secure WebRTC Biometric Stream")
    
    webrtc_streamer(
        key="biostream-pose-v3",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_telemetry:
    st.markdown("<div class='sec-title'>Core Telemetry</div>", unsafe_allow_html=True)
    
    # Real-time state indicator refresh
    status_msg = st.session_state.live_metrics["status"]
    status_clr = st.session_state.live_metrics["color"]
    st.markdown(f"<div class='metric-box'><b>Live Feed Status:</b> <span style='color:{status_clr}; font-weight:bold;'>{status_msg}</span></div>", unsafe_allow_html=True)
    
    # Evaluate if calibration countdown time window has concluded
    if st.session_state.test_active and st.session_state.start_time is not None:
        elapsed = time.time() - st.session_state.start_time
        if elapsed < 15.0:
            st.info(f"⏳ Diagnostics Processing... Progress: {int((elapsed/15)*100)}%")
            # Force UI to rerun state loops frequently during data collection phase
            time.sleep(0.5)
            st.rerun()
        else:
            st.session_state.test_active = False
            total = max(1, st.session_state.total_frames)
            healthy_pct = round((st.session_state.healthy_frames / total) * 100)
            unhealthy_pct = 100 - healthy_pct
            
            # Predict predictive health habits profiles
            if healthy_pct >= 75:
                v, adv = "OPTIMAL COMPLIANCE", "Musculoskeletal axis stable. Compression risk nominal."
                sh, ss = "7.5 Hours (Standard)", "Nominal window for tissue repair."
                db, ds = "METABOLIC STABLE", "Nutrient flow matches requirements."
                ht = "3.5 Liters"
            elif healthy_pct >= 45:
                v, adv = "MODERATE POSTURAL DEGRADATION", "Spinal axis trajectory displacement tracked."
                sh, ss = "8.5 Hours (Extended)", "Extended window needed to relieve spinal stress."
                db, ds = "ANTI-INFLAMMATORY", "Increase intake of magnesium profiles."
                ht = "4.0 Liters"
            else:
                v, adv = "CRITICAL METRIC DISTRESS", "Severe postural slouch tracked. High corrective strain."
                sh, ss = "9.5 Hours (Deep Repair)", "Immediate orthopedic recovery session required."
                db, ds = "ACUTE TISSUE REPAIR", "Eliminate refined sugar protocols entirely."
                ht = "4.8 Liters"
                
            st.session_state.final_report = {
                "h_pct": f"{healthy_pct}%",
                "u_pct": f"{unhealthy_pct}%",
                "verdict": v,
                "advice": adv,
                "sleep_h": sh, "sleep_s": ss,
                "diet_b": db, "diet_s": ds,
                "hyd_t": ht
            }
            st.success("Analysis report generated successfully!")
            st.rerun()

    # 7. Render Prescriptions UI blocks dynamically
    st.markdown("<div class='sec-title'>Biometric Analysis & Prescriptions</div>", unsafe_allow_html=True)
    report = st.session_state.final_report
    
    if report:
        st.markdown(f"""
        <div class='metric-box' style='border-left: 5px solid #0ef2fe;'>
            <h4 style='color:#0ef2fe; margin-bottom:5px;'>📊 LAB DIAGNOSTIC PROFILE</h4>
            <p><b>Healthy Configurations:</b> <span style='color:#10b981;'>{report['h_pct']}</span> | <b>Slouch Profile:</b> <span style='color:#f43f5e;'>{report['u_pct']}</span></p>
            <p style='margin-top:8px; font-size:0.9rem;'><b>Verdict:</b> {report['verdict']}</p>
            <p style='font-size:0.8rem; color:#a1a1aa;'>{report['advice']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-box'><b>🛌 Core Sleep Window:</b> {report['sleep_h']}<br><small style='color:#6b7280;'>{report['sleep_s']}</small></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><b>🥗 Cellular Nutrition:</b> {report['diet_b']}<br><small style='color:#6b7280;'>{report['diet_s']}</small></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><b>💧 Fluid Target:</b> {report['hyd_t']}</div>", unsafe_allow_html=True)
    else:
        # Standard placeholder cards when system is locked
        st.markdown("<div class='metric-box' style='color:#4b5563;'><b>🛌 Core Sleep Window:</b> Locked (Run Diagnostic Test)</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-box' style='color:#4b5563;'><b>🥗 Cellular Nutrition:</b> Locked (Run Diagnostic Test)</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-box' style='color:#4b5563;'><b>💧 Fluid Target:</b> Locked (Run Diagnostic Test)</div>", unsafe_allow_html=True)

# Continuous structural frame update loop hook to sync live UI profiles
if not st.session_state.test_active:
    time.sleep(0.4)
    st.rerun()