import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

#constants
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_EMOTIONS = len(EMOTIONS)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EMOTION_COLORS = {          # BGR for OpenCV overlays
    "Angry":    (0,   0,   220),
    "Disgust":  (0,   140,  0 ),
    "Fear":     (128,  0,  128),
    "Happy":    (0,   200,  255),
    "Sad":      (200, 100,   0 ),
    "Surprise": (0,   165,  255),
    "Neutral":  (180, 180,  180),
}

class ChannelAttenion(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w
    
#classifier head

class EmotionHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
    

