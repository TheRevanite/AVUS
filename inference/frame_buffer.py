import torch

class FrameBuffer:
    def __init__(self, max_length=16):
        self.buffer = []
        self.max_length = max_length

    def add_frame(self, frame_tensor):
        self.buffer.append(frame_tensor)
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)

    def is_ready(self):
        return len(self.buffer) == self.max_length

    def get_clip_tensor(self):
        # shape: (1, 3, 16, H, W)
        return torch.stack(self.buffer).permute(1, 0, 2, 3).unsqueeze(0)
