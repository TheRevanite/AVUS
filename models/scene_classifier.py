import torch
import torch.nn as nn
import torchvision.models as models
import urllib.request
import os

class SceneClassifier(nn.Module):
    def __init__(self, num_classes=365):
        super().__init__()
        
        self.model = models.resnet50(num_classes=num_classes)

        weight_url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
        weight_path = "resnet50_places365.pth.tar"
        if not os.path.exists(weight_path):
            print("Downloading Places365 model...")
            urllib.request.urlretrieve(weight_url, weight_path)
        
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.class_names = self.load_places365_labels(num_classes)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

    @staticmethod
    def load_places365_labels(num_classes):
        label_url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
        with urllib.request.urlopen(label_url) as f:
            lines = f.read().decode("utf-8").splitlines()
        
        labels = [line.strip().split(' ')[0][3:] for line in lines if line.strip()]

        return [labels[i] if i < len(labels) else f"Scene {i}" for i in range(num_classes)]
