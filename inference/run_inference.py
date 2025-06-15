import torch
import torchvision.transforms as transforms
import cv2

def get_transforms():
    transform_scene = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_action = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                             std=[0.22803, 0.22145, 0.216989])
    ])
    return transform_scene, transform_action

def run_inference_on_frame(frame, frame_buffer, models, transforms):
    action_model, scene_model, object_detector = models
    transform_scene, transform_action = transforms

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== ACTION =====
    frame_tensor = transform_action(rgb)
    frame_buffer.add_frame(frame_tensor)

    if frame_buffer.is_ready():
        clip_tensor = frame_buffer.get_clip_tensor()  # (1, 3, 16, H, W)
        with torch.no_grad():
            action_logits = action_model(clip_tensor)
        action_id = action_logits.argmax().item()
        action_label = action_model.class_names[action_id]
    else:
        action_label = "Loading..."

    # ===== SCENE =====
    scene_input = transform_scene(rgb).unsqueeze(0)
    with torch.no_grad():
        scene_logits = scene_model(scene_input)
    scene_id = scene_logits.argmax().item()
    scene_label = scene_model.class_names[scene_id]

    # ===== OBJECT DETECTION =====
    detections = object_detector.detect(frame)

    return action_label, scene_label, detections
