import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    卷积块：Conv2d -> BatchNorm -> ReLU -> (可选)MaxPool
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, use_pool=True):
        super(ConvBlock, self).__init__()

        self.use_pool = use_pool

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # BatchNorm后面不需要bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if self.use_pool:
            x = self.pool(x)
        return x


class CNN5Layer(nn.Module):
    """
    5层卷积神经网络

    架构设计:
        输入: 3 x 224 x 224
        Conv1: 3->32,  BN, ReLU, MaxPool -> 32 x 112 x 112
        Conv2: 32->64,  BN, ReLU, MaxPool -> 64 x 56 x 56
        Conv3: 64->128, BN, ReLU, MaxPool -> 128 x 28 x 28
        Conv4: 128->256,BN, ReLU, MaxPool -> 256 x 14 x 14
        Conv5: 256->512,BN, ReLU, MaxPool -> 512 x 7 x 7
        GlobalAvgPool -> 512
        Dropout(0.5)
        FC: 512 -> 256 -> ReLU -> Dropout(0.3)
        FC: 256 -> 2 (cat/dog)
    """

    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(CNN5Layer, self).__init__()

        # ========== 特征提取部分 (5层卷积) ==========
        self.conv1 = ConvBlock(3, 32, kernel_size=3, padding=1)
        # 输出: 32 x 112 x 112

        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1)
        # 输出: 64 x 56 x 56

        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1)
        # 输出: 128 x 28 x 28

        self.conv4 = ConvBlock(128, 256, kernel_size=3, padding=1)
        # 输出: 256 x 14 x 14

        self.conv5 = ConvBlock(256, 512, kernel_size=3, padding=1)
        # 输出: 512 x 7 x 7

        # ========== 全局平均池化 ==========
        # 将 7x7 的特征图池化为 1x1，减少参数量
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ========== 分类器部分 ==========
        self.classifier = nn.Sequential(
            # Dropout层：训练时随机丢弃50%神经元，防止过拟合
            nn.Dropout(dropout_rate),

            # 全连接层1：512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # 全连接层也可以用BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 较低的Dropout率

            # 全连接层2：256 -> num_classes
            nn.Linear(256, num_classes)
        )

        # ========== 权重初始化 ==========
        self._initialize_weights()

    def _initialize_weights(self):
        """使用Kaiming初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 5层卷积特征提取
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # [B, 512, 7, 7]

        # 全局平均池化 -> [B, 512, 1, 1]
        x = self.global_avg_pool(x)

        # 展平 -> [B, 512]
        x = torch.flatten(x, 1)

        # 分类器
        x = self.classifier(x)

        return x

    def get_features(self, x):
        """获取特征图（用于Grad-CAM可视化）"""
        features = []
        x = self.conv1(x);
        features.append(x)
        x = self.conv2(x);
        features.append(x)
        x = self.conv3(x);
        features.append(x)
        x = self.conv4(x);
        features.append(x)
        x = self.conv5(x);
        features.append(x)
        return features


# ==================== 测试模型 ====================

def test_model():
    """测试模型结构和前向传播"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = CNN5Layer(num_classes=2, dropout_rate=0.5).to(device)

    # 打印模型结构
    print("=" * 60)
    print("模型结构：")
    print(model)
    print("=" * 60)

    # 使用 torchsummary 查看每层参数（仅测试脚本需要，可选依赖）
    try:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device=str(device))
    except ImportError:
        print('提示: 安装 torchsummary 可查看更详细的层信息 (pip install torchsummary)')
    except Exception:
        pass

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")  # 期望: [4, 2]
    print(f"输出值: {output}")

    return model


if __name__ == '__main__':
    model = test_model()