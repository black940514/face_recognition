import torch
import torch.nn as nn
import timm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.nn.functional import softmax

'''
    이미지 특징 추출 모델
'''
class ImageFeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super(ImageFeatureExtractor, self).__init__()
        # Load pre-trained ResNet-18 from timm
        self.resnet = timm.create_model('resnet18', pretrained=True, num_classes=0)
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.resnet(x)
        features = self.fc(features)
        # return features.view(x.shape[0], -1)
        return features


class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim, num_heads, num_classes, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        # 트랜스포머의 인코더 레이어 (어텐션 적용)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.frame_weights = nn.Linear(feature_dim, 1)  # Weight for each frame

        # 분류 레이어
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, src):
        # 입력(src)의 형태: (프레임 개수, 배치 크기, feature vector 차원수)
        # 인코딩 시작
        transformed = self.transformer_encoder(src)

        # Calculate attention weights and take weighted average
        weights = softmax(self.frame_weights(transformed), dim=0)
        weighted_transformed = (transformed * weights).sum(dim=0)

        output = self.classifier(weighted_transformed)
        return output

class ImageTransformerClassifier(nn.Module):
    def __init__(self, num_images, num_classes):
        super(ImageTransformerClassifier, self).__init__()
        self.feature_extractor = ImageFeatureExtractor(feature_dim=128)
        self.encoder = TransformerEncoder(feature_dim=128, num_heads=1, num_classes=num_classes)

    def forward(self, x):
        # x shape: (batch_size, num_images, C, H, W)
        batch_size, num_images, C, H, W = x.size()
        x = x.view(-1, C, H, W)  # Flatten to (batch_size * num_images, C, H, W)

        # Extract features
        features = self.feature_extractor(x)  # (batch_size * num_images, 128)
        features = features.view(batch_size, num_images, -1).transpose(0, 1)  # Prepare for transformer (num_images, batch_size, 128)

        # Classify using transformer encoder
        output = self.encoder(features)
        return output