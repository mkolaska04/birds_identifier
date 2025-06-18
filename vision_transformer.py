import torch
import torch.nn as nn
import timm 
from tqdm import tqdm
import os
from transfer_learning import train_cycle, evaluate_model 
from utils.metrics_plotting import plot_learning_curves, plot_confusion_matrix_and_report

def get_vit_model(num_classes, pretrained=True):
    """Loads a pretrained Vision Transformer model."""
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    return model

def run_vision_transformer(num_classes, train_loader, val_loader, test_loader, class_names, device):
    """Main function to run fine-tuning for Vision Transformer."""
    model_name = "VisionTransformer"
    print(f"\n{'='*20} Running Fine-Tuning for {model_name} {'='*20}")
    
    model = get_vit_model(num_classes)
    
    for param in model.parameters():
        param.requires_grad = True
        
    model, history = train_cycle(model, model_name, train_loader, val_loader, epochs=10, lr=1e-5, device=device)
    
    plot_learning_curves(history, model_name)
    
    model_path = os.path.join("results", f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Final Vision Transformer model saved to {model_path}")

    y_true, y_pred = evaluate_model(model, test_loader, device)
    plot_confusion_matrix_and_report(y_true, y_pred, class_names, model_name)
    
    print(f"\n{'='*20} Finished {model_name} {'='*20}")