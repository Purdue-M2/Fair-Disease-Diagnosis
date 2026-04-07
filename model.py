"""3D ResNet-18 for lung disease diagnosis (grayscale CT input)."""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class LungDiagnosisModel3D(nn.Module):
    """3D ResNet-18 adapted for single-channel CT volumes.

    Loads Kinetics-400 pretrained weights and converts the first convolution
    from 3-channel RGB to 1-channel grayscale by averaging.

    Args:
        num_classes: Number of disease categories (default: 4).
    """

    def __init__(self, num_classes=4):
        super().__init__()
        base_model = r3d_18(weights=R3D_18_Weights.DEFAULT)

        old_conv = base_model.stem[0]
        new_conv = nn.Conv3d(
            1, 64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        base_model.stem[0] = new_conv

        self.features = nn.Sequential(
            base_model.stem,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
