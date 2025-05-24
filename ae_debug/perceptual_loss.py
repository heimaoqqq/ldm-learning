import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:16])  # relu_2_2
        for param in self.layers.parameters():
            param.requires_grad = False
        self.resize = resize
    def forward(self, x, y):
        # x, y: [B, 3, H, W], 归一化到[0,1]
        if self.resize:
            x = nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224,224), mode='bilinear', align_corners=False)
        return nn.functional.l1_loss(self.layers(x), self.layers(y)) 