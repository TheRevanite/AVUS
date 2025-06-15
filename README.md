# AVUS: Action, Scene, and Object Understanding from Video

AVUS is a Python-based pipeline for real-time video understanding. It performs action recognition, scene classification, and object detection on video streams.

## Features

- **Action Recognition:** Uses a 3D ResNet model trained on Kinetics.
- **Scene Classification:** Uses ResNet50 trained on Places365.
- **Object Detection:** Uses YOLOv8 via Ultralytics.

## Project Structure

```
main.py
config.yaml
requirements.txt
models/
inference/
setup/
utils/
demo/
data/
```

## Getting Started

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Prepare Model Weights

- Download `resnet50_places365.pth.tar` and `yolo11n.pt` if not already present.
- The action and scene models will download labels automatically.

### 3. Configure

Edit `config.yaml` to set paths and parameters, e.g.:

```yaml
video_path: "path/to/video.mp4"
action_num_classes: 400
scene_num_classes: 365
object_detector_model_path: "yolo11n.pt"
```

### 4. Run the Main Pipeline

```sh
python main.py
```

### 5. Run the Demo (Streamlit)

```sh
streamlit run demo/app.py
```

## Credits

- Action model: Kinetics-400
- Scene model: Places365
- Object detection: YOLOv11 (Ultralytics)

## License

MIT License
