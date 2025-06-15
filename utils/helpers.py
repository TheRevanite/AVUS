import cv2
from collections import Counter

def draw_labels(frame, action_label, scene_label):
    cv2.putText(frame, f"Action: {action_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Scene: {scene_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

def draw_detections(frame, detections, class_names):
    if detections is not None and hasattr(detections, 'boxes'):
        class_ids = [int(box.cls[0]) for box in detections.boxes]
        counts = Counter(class_ids)
        y_offset = 110
        for cls_id, count in counts.items():
            label_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"Class {cls_id}"
            label = f"{label_name}: {count}"
            cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{cls} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
