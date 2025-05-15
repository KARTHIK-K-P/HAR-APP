import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load model
MODEL_PATH = "output/pose_classifier.pkl"
model = joblib.load(MODEL_PATH)

def extract_landmarks(results):
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in landmarks:
        data.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(data).reshape(1, -1)

def process_frame(img, pose):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        features = extract_landmarks(results)
        if features is not None:
            prediction = model.predict(features)[0]
            cv2.putText(img, f'Pose: {prediction}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        else:
            cv2.putText(img, "No Pose Detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    else:
        cv2.putText(img, "No Pose Detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    return img

st.title("Pose Classification: Webcam & Stored Videos ")

option = st.sidebar.selectbox("Select Input Source", ["Webcam", "Stored Videos"])

if 'running' not in st.session_state:
    st.session_state.running = False

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
stframe = st.empty()

video_folder = "videos"
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")] if option == "Stored Videos" else []

selected_video = None
if option == "Stored Videos":
    selected_video = st.sidebar.selectbox("Select a Video", video_files)
    video_path = os.path.join(video_folder, selected_video)

# Buttons to control Start/Stop
col1, col2 = st.columns(2)
with col1:
    start = st.button("Start")
with col2:
    stop = st.button("Stop")

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# Main loop managed by st.empty() and timer
if st.session_state.running:
    if option == "Stored Videos":
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open video source")
        st.session_state.running = False
    else:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                if option == "Stored Videos":
                    # Loop video if ended
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    st.warning("Unable to read frame.")
                    break

            # Resize frame to 640x480 for consistency (you can adjust size)
            frame = cv2.resize(frame, (640, 480))

            frame = process_frame(frame, pose)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

            # Short delay to avoid overloading CPU & allow button press detection
            time.sleep(0.03)

    cap.release()
else:
    stframe.empty()

pose.close()
