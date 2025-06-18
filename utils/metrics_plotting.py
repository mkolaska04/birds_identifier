# utils/metrics_plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import torch

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def _unnormalize_for_display(tensor_img):
    """Un-normalizes a tensor image for display with matplotlib."""
    inp = tensor_img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def show_augmentation_examples(dataset, class_names, num_examples=5):
    """
    Loads a few images from the dataset, displays the original,
    and displays its augmented version.
    """
    fig = plt.figure(figsize=(10, num_examples * 3))
    plt.suptitle("Examples of Training Data Augmentation", fontsize=16)

    for i in range(num_examples):
        # The dataset __getitem__ returns the augmented tensor
        augmented_tensor, label_idx = dataset[i]
        
        # To show the original, we need to load it from disk without augmentation
        img_info = dataset.df.iloc[i]
        original_img_path = os.path.join(dataset.dataset_path, 'images', img_info['filepath'])
        original_pil_img = Image.open(original_img_path).convert('RGB')
        
        class_name = class_names[label_idx]

        # Plot original image
        ax = plt.subplot(num_examples, 2, 2 * i + 1)
        plt.imshow(original_pil_img)
        ax.set_title(f"Original (Class: {class_name})")
        plt.axis('off')

        # Plot augmented image
        ax = plt.subplot(num_examples, 2, 2 * i + 2)
        augmented_img_display = _unnormalize_for_display(augmented_tensor)
        plt.imshow(augmented_img_display)
        ax.set_title(f"Augmented (Class: {class_name})")
        plt.axis('off')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_learning_curves(history, model_name):
    """Plots and saves learning curves for accuracy and loss."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Accuracy for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'learning_curves_{model_name}.png')
    plt.savefig(save_path)
    print(f"Learning curves saved to {save_path}")
    plt.show()


def plot_confusion_matrix_and_report(y_true, y_pred, class_names, model_name):
    """Generates, saves, and prints classification report and confusion matrix."""
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    report_path = os.path.join(RESULTS_DIR, f'report_{model_name}.csv')
    df_report.to_csv(report_path)
    print(f"\n--- Final Test Report for {model_name} ---")
    print(df_report)
    print(f"Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', xticklabels=False, yticklabels=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.show()