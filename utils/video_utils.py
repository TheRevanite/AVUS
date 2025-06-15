import cv2
from setup.video_loader import load_video
from inference.run_inference import run_inference_on_frame
from utils.helpers import draw_labels, draw_detections


def process_video(video_path, process_frame_callback):
    cap = load_video(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame_callback(frame)

        cv2.namedWindow("AVUS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AVUS", 1280, 720)

        resized_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("AVUS", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def handle_frame_factory(frame_buffer, models, transforms):
    def handle_frame(frame):
        action_label, scene_label, detections = run_inference_on_frame(
            frame, frame_buffer, models, transforms
        )
        draw_labels(frame, action_label, scene_label)
        draw_detections(frame, detections, models[2].class_names)
    return handle_frame
