from setup.model_loader import load_models
from inference.run_inference import get_transforms
from utils.video_utils import process_video, handle_frame_factory
from inference.frame_buffer import FrameBuffer
from config.config import load_config

def main(config):
    models = load_models(config['action_num_classes'], config['scene_num_classes'], config['object_detector_model_path'])
    transforms = get_transforms()
    frame_buffer = FrameBuffer(max_length=16)
    handle_frame = handle_frame_factory(frame_buffer, models, transforms)
    process_video(config['video_path'], handle_frame)

if __name__ == "__main__":
    config = load_config()
    main(config)