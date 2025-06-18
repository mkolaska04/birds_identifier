import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import os
from utils.metrics_plotting import plot_learning_curves, plot_confusion_matrix_and_report

def create_optimizer_with_differential_lr(model, model_name, base_lr):
    params_to_optimize = []
    
    if "ResNet" in model_name:
        head_params = list(model.fc.parameters())
        layer4_params = list(model.layer4.parameters())
        base_params = [p for n, p in model.named_parameters() if "fc" not in n and "layer4" not in n]
        params_to_optimize = [
            {'params': base_params, 'lr': base_lr / 20},
            {'params': layer4_params, 'lr': base_lr / 4},
            {'params': head_params, 'lr': base_lr}
        ]
    elif "MobileNet" in model_name or "EfficientNet" in model_name:
        head_params = list(model.classifier.parameters())
        base_params = [p for n, p in model.named_parameters() if "classifier" not in n]
        params_to_optimize = [
            {'params': base_params, 'lr': base_lr / 10},
            {'params': head_params, 'lr': base_lr}
        ]
    else:
        params_to_optimize = model.parameters()

    return optim.AdamW(params_to_optimize, lr=base_lr, weight_decay=1e-2)


def train_cycle(model, model_id, train_loader, val_loader, epochs, lr, device, weight_decay=1e-2, patience=5, use_differential_lr=False):
    """
    Runs a full training and validation cycle with Early Stopping.
    Accepts weight_decay, patience, and use_differential_lr as parameters.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if use_differential_lr and "finetuned" in model_id:
        print("Using differential learning rates for fine-tuning.")
        optimizer = create_optimizer_with_differential_lr(model, model_id, base_lr=lr)
    else:
        print("Using standard optimizer with a single learning rate.")
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs("results", exist_ok=True)
    best_model_path = os.path.join("results", f"best_model_{model_id}.pth")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f} | Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        scheduler.step()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Validation loss improved to {best_val_loss:.4f}. Saving model to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break
            
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("No best model was saved (training might have been too short or validation loss never improved).")
    
    return model, history

def evaluate_model(model, test_loader, device):
    """Gets predictions from a model for the test set."""
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

def run_transfer_learning(model_name, num_classes, train_loader, val_loader, test_loader, class_names, device):
    print(f"\n{'='*20} Running Transfer Learning for {model_name} {'='*20}")

    model_map = {
        "MobileNetV2": (models.mobilenet_v2, "classifier"),
        "ResNet50": (models.resnet50, "fc"),
        "EfficientNetB0": (models.efficientnet_b0, "classifier")
    }
    model_constructor, classifier_name = model_map[model_name]

    print("\n--- Phase 1: Training Classifier (Frozen Backbone) ---")
    model = model_constructor(weights='DEFAULT')
    
    if classifier_name == 'classifier':
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else: 
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    for param in model.parameters(): param.requires_grad = False
    for param in getattr(model, classifier_name).parameters(): param.requires_grad = True

    model_id_frozen = f"{model_name}_frozen"
    model, history_frozen = train_cycle(model, model_id_frozen, train_loader, val_loader, epochs=5, lr=0.001, device=device)
    
    plot_learning_curves(history_frozen, model_id_frozen)
    y_true, y_pred = evaluate_model(model, test_loader, device)
    plot_confusion_matrix_and_report(y_true, y_pred, class_names, model_id_frozen)

    print("\n--- Phase 2: Fine-Tuning (Unfrozen Backbone) ---")
    for param in model.parameters(): param.requires_grad = True
    
    model_id_finetuned = f"{model_name}_finetuned"
    model, history_finetune = train_cycle(
        model, model_id_finetuned, train_loader, val_loader, 
        epochs=20, lr=1e-4, device=device, patience=5, use_differential_lr=True
    )
    
    plot_learning_curves(history_finetune, model_id_finetuned)
    y_true, y_pred = evaluate_model(model, test_loader, device)
    plot_confusion_matrix_and_report(y_true, y_pred, class_names, model_id_finetuned)
    
    print(f"\n{'='*20} Finished {model_name} {'='*20}")
