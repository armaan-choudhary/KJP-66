import torch
import torch.nn as nn
import time

# PrismNet Project: Early-Exit Classification (GB-03 Innovation)
print("--- PrismNet: Early-Exit Classification System ---")

class EarlyExitHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EarlyExitHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class EarlyExitResNet(nn.Module):
    """
    Dynamically compresses inference depth based on input complexity.
    Optimized for ImageNet (1000 classes).
    """
    def __init__(self, base_model, num_classes=1000, threshold=0.85):
        super(EarlyExitResNet, self).__init__()
        self.threshold = threshold
        
        # Sub-modules from ResNet-50
        self.stem = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.layer1 = base_model.layer1 # 256
        self.layer2 = base_model.layer2 # 512
        self.layer3 = base_model.layer3 # 1024
        self.layer4 = base_model.layer4 # 2048
        
        # Branch Heads
        self.exit1 = EarlyExitHead(256, num_classes)
        self.exit2 = EarlyExitHead(512, num_classes)
        self.final_exit = nn.Sequential(base_model.avgpool, nn.Flatten(), base_model.fc)
        
    def forward(self, x):
        # Stage 0: Stem
        x = self.stem(x)
        
        # Stage 1: Layer 1
        x = self.layer1(x)
        logits1 = self.exit1(x)
        probs1 = torch.softmax(logits1, dim=1)
        conf1, _ = torch.max(probs1, 1)
        
        if not self.training and conf1.item() >= self.threshold:
            return logits1, 1
            
        # Stage 2: Layer 2
        x = self.layer2(x)
        logits2 = self.exit2(x)
        probs2 = torch.softmax(logits2, dim=1)
        conf2, _ = torch.max(probs2, 1)
        
        if not self.training and conf2.item() >= self.threshold:
            return logits2, 2
            
        # Stage 3: Full Depth
        x = self.layer3(x)
        x = self.layer4(x)
        logits_final = self.final_exit(x)
        return logits_final, 3
