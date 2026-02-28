import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from baseline import get_resnet50_places365
import os

# PrismNet Project: Early-Exit Inference (Dynamic Depth)
print("--- PrismNet: Early-Exit System (Dynamic Depth) ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    EarlyExitResNet: Confidence-aware dynamic depth.
    Exits:
    - Exit 1: After layer1 (256 channels)
    - Exit 2: After layer2 (512 channels)
    - Final Exit: After layer4 (2048 channels)
    """
    def __init__(self, base_model, num_classes=365, threshold=0.85):
        super(EarlyExitResNet, self).__init__()
        self.threshold = threshold
        self.num_classes = num_classes
        
        # Extract layers from base_model
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1 # 256 ch
        self.layer2 = base_model.layer2 # 512 ch
        self.layer3 = base_model.layer3 # 1024 ch
        self.layer4 = base_model.layer4 # 2048 ch
        
        # Exit Heads
        self.exit1 = EarlyExitHead(256, num_classes)
        self.exit2 = EarlyExitHead(512, num_classes)
        self.final_exit = nn.Sequential(
            base_model.avgpool,
            nn.Flatten(),
            base_model.fc
        )
        
    def forward(self, x):
        # Common prefix
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layer 1 -> Exit 1
        x = self.layer1(x)
        logits1 = self.exit1(x)
        probs1 = torch.nn.functional.softmax(logits1, dim=1)
        conf1, _ = torch.max(probs1, 1)
        
        if not self.training and conf1.item() >= self.threshold:
            return logits1, 1
            
        # Layer 2 -> Exit 2
        x = self.layer2(x)
        logits2 = self.exit2(x)
        probs2 = torch.nn.functional.softmax(logits2, dim=1)
        conf2, _ = torch.max(probs2, 1)
        
        if not self.training and conf2.item() >= self.threshold:
            return logits2, 2
            
        # Layer 3 & 4 -> Final Exit
        x = self.layer3(x)
        x = self.layer4(x)
        logits_final = self.final_exit(x)
        
        return logits_final, 3

def fine_tune_exits(model, epochs=2):
    """
    Fine-tunes the early-exit heads while freezing the backbone.
    """
    print(f"Fine-tuning exit heads for {epochs} epochs...")
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze exit heads
    for param in model.exit1.parameters():
        param.requires_grad = True
    for param in model.exit2.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Synthetic calibration data (Places365 not available)
    # 8 samples, 3 channels, 224x224
    dummy_loader = [(torch.randn(8, 3, 224, 224).to(device), torch.randint(0, 365, (8,)).to(device))]
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dummy_loader:
            optimizer.zero_grad()
            
            # During training, we might want to train all exits
            # This is a simplified training loop
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            
            # Exit 1 loss
            x = model.layer1(x)
            out1 = model.exit1(x)
            loss1 = criterion(out1, labels)
            
            # Exit 2 loss
            x = model.layer2(x)
            out2 = model.exit2(x)
            loss2 = criterion(out2, labels)
            
            total_loss = loss1 + loss2
            total_loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f}")
        
    # Save early-exit model
    torch.save(model.state_dict(), 'early_exit_model.pth')
    print("Fine-tuning complete. Model saved to early_exit_model.pth")
    return model

if __name__ == "__main__":
    base = get_resnet50_places365(pretrained=False)
    ee_model = EarlyExitResNet(base).to(device)
    fine_tune_exits(ee_model)
