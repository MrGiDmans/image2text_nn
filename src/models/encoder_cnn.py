import torch.nn as nn
import torchvision.models as models
from torch import Tensor

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14, embed_dim=512, fine_tune=False):
        """
        encoded_image_size: the size of the feature map (for attention, if necessary)
        embed_dim: size of the output feature vector
        fine_tune: allow/prohibit training of conv layers
        """
        super().__init__()
        self.enc_image_size = encoded_image_size

        # берем предобученный ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # убираем fully connected слой и avgpool
        modules = list(resnet.children())[:-2]  # оставляем conv блоки (последние два dims)
        self.backbone = nn.Sequential(*modules)

        # адаптивный pooling чтобы получить фиксированный spatial размер
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # проекция каналов ResNet -> embed_dim
        # ResNet-50 out channels = 2048
        self.conv = nn.Conv2d(2048, embed_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fine_tune(fine_tune)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        images: FloatTensor [B, 3, H, W] (pre-normalized)
        returns:
         - features: [B, num_pixels, embed_dim] where num_pixels = self.enc_image_size**2
         - global_features: [B, embed_dim] (mean-pooled)
        """
        x: Tensor = self.backbone(images)            # [B, 2048, H', W']
        x = self.adaptive_pool(x)                    # [B, 2048, enc_image_size, enc_image_size]
        x = self.conv(x)                             # [B, embed_dim, enc_image_size, enc_image_size]
        x = self.relu(x)
        b, c, h, w = x.size()
        features = x.view(b, c, h * w).permute(0, 2, 1)  # [B, num_pixels, embed_dim]
        global_feat = features.mean(dim=1)               # [B, embed_dim]
        return features, global_feat

    def fine_tune(self, fine_tune: bool = False, trainable_blocks: int = 2) -> None:
        """
        Размораживаем только последние `trainable_blocks` резнет-блока, чтобы
        избежать «забывания» базовых признаков.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False

        if fine_tune:
            blocks = list(self.backbone.children())
            for block in blocks[-trainable_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
