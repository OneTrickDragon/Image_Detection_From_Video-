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
    

class EmotionNet(nn.Module):
    def __init__(self, pretrained=True, freeze_bn=False, dropout = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b2", pretrained=pretrained, nums_classes = 0, global_pool = ""
        )
        feat_dim = self.backbone.num_features #1408
        self.attention = ChannelAttenion(feat_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = EmotionHead(feat_dim, NUM_EMOTIONS, dropout)
        if freeze_bn: self._freeze_bn()

    
    def _freeze_bn(self):
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                for p in m.parameters(): p.requires_grad = False

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self, layers_from_end = 3):
        for p in self.backbone.parameters(): p.requires_grad = True
        for block in list(self.backbone.children())[-layers_from_end]:
            for p in self.backbone.parameters(): p.requires_grad = False

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        feats = self.attention(feats)
        feats = self.flatten(self.pool(feats))

        return self.classifier(feats)
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        probs = F.softmax(self(x), dim=-1)[0]
        idx = probs.argmax().item()
        return {
            "label": EMOTIONS[idx],
            "confidence": probs[idx].item(),
            "probabilities": {e: probs[i].item() for i,e in enumerate(EMOTIONS)}
        }