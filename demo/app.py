import streamlit as st
import os
from preprocessing.video_preprocessing import load_and_preprocess_frames
from models.feature_extractor import SharedBackbone
from models.action_model import ActionRecognitionHead
from models.scene_classifier import SceneClassifier
from models.object_detector import ObjectDetector
from inference.run_inference import predict

st.title("🎥 Multi-Task Video Understanding System")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if video_file:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.info("[1/4] Preprocessing frames...")
    frames = load_and_preprocess_frames(video_path)
    st.success(f"Loaded {len(frames)} frames.")

    st.info("[2/4] Loading models...")
    backbone = SharedBackbone()
    action_model = ActionRecognitionHead(num_classes=10)
    scene_model = SceneClassifier(num_classes=365)
    object_detector = ObjectDetector(model_path='yolov5s.pt')

    backbone.eval()
    action_model.eval()
    scene_model.eval()
    st.success("Models loaded.")

    st.info("[3/4] Running inference...")
    results = predict(frames, backbone, action_model, scene_model, object_detector)
    st.success("Inference complete!")

    st.header("🎯 Inference Results")
    st.subheader("Action Predictions")
    st.write(results['actions'])

    st.subheader("Scene Predictions")
    st.write(results['scenes'])

    st.subheader("Object Detections (first frame)")
    st.write(results['objects'][0])
