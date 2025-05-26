import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        # 使用 VGG16
        self.vgg = models.vgg16(pretrained=True).features
        self.vgg.eval()
        
        # 定义需要提取特征的层索引
        self.layer_indices = {'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15, 'relu4_3': 22}
        
        # 将特征提取层设置为不需要梯度
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.resize = resize
        
    def to(self, device):
        # 确保VGG模型也被移到正确的设备上
        self.vgg = self.vgg.to(device)
        return super().to(device)

    def forward(self, x, y):
        # x, y: [B, 3, H, W], 归一化到[0,1] (通常感知损失前需要这样处理)
        if x.shape[1] == 1: # 如果是单通道图像，复制到3通道
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        if self.resize:
            x = nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224,224), mode='bilinear', align_corners=False)
        
        # 提取特征并计算损失
        loss = 0.0
        
        def get_features(image, model, layers_indices):
            features = []
            temp_image = image
            for i, layer in enumerate(model):
                temp_image = layer(temp_image)
                if i in layers_indices:
                    features.append(temp_image)
            return features

        target_layer_indices_list = sorted(self.layer_indices.values())
        
        features_x = get_features(x, self.vgg, target_layer_indices_list)
        features_y = get_features(y, self.vgg, target_layer_indices_list)

        for feat_x_layer, feat_y_layer in zip(features_x, features_y):
            loss += nn.functional.l1_loss(feat_x_layer, feat_y_layer)
            
        return loss 