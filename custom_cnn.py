import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transfer_learning import train_cycle, evaluate_model
from utils.metrics_plotting import plot_learning_curves, plot_confusion_matrix_and_report

class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity 
        out = self.relu(out)
        return out

class FinalCNN(nn.Module):
    """A custom CNN inspired by ResNet architecture."""
    def __init__(self, num_classes):
        super(FinalCNN, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResidualBlock, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, num_blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def find_lr(model, train_loader, criterion, device):
    """Performs a learning rate range test."""
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
    lr_finder = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.1)
    
    losses = []
    lrs = []
    
    model.train()
    model.to(device)
    
    for inputs, labels in tqdm(train_loader, desc="Finding best LR"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss.item())
        
        lr_finder.step()
        if loss.item() > 4 * (min(losses) if losses else 1.0) or torch.isnan(loss):
            print("Loss exploded. Stopping LR finder.")
            break

    plt.figure(figsize=(10, 5))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Range Test")
    plt.grid(True)
    plt.show()
    
    print("Znajdź na wykresie punkt, gdzie strata spada najszybciej i odczytaj 'learning rate' z osi X.")
    print("Dobre wartości to często 1e-4, 1e-3, 5e-3.")


def run_custom_cnn(num_classes, train_loader, val_loader, test_loader, class_names, device, lr=1e-3):
    """Trains and evaluates the custom CNN with a specified learning rate."""
    model_id = "FinalCustomCNN"
    print(f"\n{'='*20} Running {model_id} with lr={lr} {'='*20}")
    
    model = FinalCNN(num_classes)
    
    model, history = train_cycle(
        model, model_id, train_loader, val_loader, 
        epochs=60, lr=lr, device=device, weight_decay=1e-4, patience=7 
    )
    
    plot_learning_curves(history, model_id)
    y_true, y_pred = evaluate_model(model, test_loader, device)
    plot_confusion_matrix_and_report(y_true, y_pred, class_names, model_id)
    
    model_path = os.path.join("results", f"{model_id}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Custom CNN model saved to {model_path}")
    print(f"\n{'='*20} Finished {model_id} {'='*20}")