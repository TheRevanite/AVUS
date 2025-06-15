from models.action_model import ActionRecognitionHead
from models.scene_classifier import SceneClassifier
from models.object_detector import ObjectDetector

def load_models(action_classes, scene_classes, object_model_path):
    print("[1/3] Loading models...")
    action_model = ActionRecognitionHead(num_classes=action_classes).eval()
    scene_model = SceneClassifier(num_classes=scene_classes).eval()
    object_detector = ObjectDetector(model_path=object_model_path)
    return action_model, scene_model, object_detector

