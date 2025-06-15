from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolo11n.pt'):
        self.model = YOLO(model_path)
        self.class_names = self.model.model.names

    def detect(self, frame):
        return self.model(frame)[0]
