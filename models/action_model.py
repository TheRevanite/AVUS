import torch
import torch.nn as nn
import torchvision.models.video as video_models
import urllib.request

class ActionRecognitionHead(nn.Module):
    def __init__(self, num_classes=400):
        super().__init__()
        self.model = video_models.r3d_18(pretrained=True)
        self.model.eval()
        self.class_names = self.load_kinetics_labels(num_classes)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

    @staticmethod
    def load_kinetics_labels(num_classes):
        url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with urllib.request.urlopen(url) as f:
            labels = f.read().decode("utf-8").splitlines()

        labels = [label.strip() for label in labels if label.strip()]
        print("✅ Loaded label count:", len(labels))
        print("🎯 Sample labels:", labels[:5])

        # Pad with "Class i" if labels are missing
        return [labels[i] if i < len(labels) else f"Class {i}" for i in range(num_classes)]
