import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
from config.config import load_config
from setup.model_loader import load_models
from inference.run_inference import run_inference_on_frame, get_transforms
from inference.frame_buffer import FrameBuffer
from utils.helpers import draw_labels, draw_detections

st.title("AVUS Video Understanding Demo")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    config = load_config()
    models = load_models(config['action_num_classes'], config['scene_num_classes'], config['object_detector_model_path'])
    transforms = get_transforms()
    frame_buffer = FrameBuffer(max_length=16)

    tfile = open("temp_video.mp4", "wb")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture("temp_video.mp4")
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        action_label, scene_label, detections = run_inference_on_frame(frame, frame_buffer, models, transforms)
        draw_labels(frame, action_label, scene_label)
        draw_detections(frame, detections, models[2].class_names)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    cap.release()