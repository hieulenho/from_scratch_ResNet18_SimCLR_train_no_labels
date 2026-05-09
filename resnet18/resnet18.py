import torch
import torch.nn as nn
from .basicblock import BasicBlock, Bottleneck


class ResNet(nn.Module):
    """ResNet backbone/classifier with an option to return pooled features.

    Supported factory functions: resnet18, resnet34, resnet50.
    - If return_features=True: returns pooled features [B, feature_dim].
    - Else: returns logits [B, num_classes].
    """

    def __init__(self, block, layers, num_classes: int = 10, zero_init_residual: bool = False):
        super().__init__()
        self.in_channels = 64
        self.num_classes = num_classes
        self.feature_dim = 512 * block.expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_dim, num_classes) if num_classes is not None else nn.Identity()

        self._init_weights(zero_init_residual=zero_init_residual)

    def _make_layer(self, block, out_channels: int, blocks: int, stride: int):
        downsample = None
        out_expanded = out_channels * block.expansion
        if stride != 1 or self.in_channels != out_expanded:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_expanded, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_expanded),
            )

        layers = [block(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_expanded
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        if return_features:
            return feats
        return self.fc(feats)


def resnet18(num_classes: int = 10) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes: int = 10) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes: int = 10) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


ARCH_BUILDERS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


def build_resnet(arch: str, num_classes: int = 10) -> ResNet:
    arch = arch.lower().strip()
    if arch not in ARCH_BUILDERS:
        raise ValueError(f"Unsupported arch '{arch}'. Choose one of: {sorted(ARCH_BUILDERS)}")
    return ARCH_BUILDERS[arch](num_classes=num_classes)
