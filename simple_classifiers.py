
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV 
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from utils.metrics_plotting import plot_confusion_matrix_and_report

def extract_features(dataloader, model, device):
    """Extracts features and labels from a dataloader using a model."""
    features, labels = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            features.append(outputs)
            labels.append(targets.numpy())
    return np.vstack(features), np.hstack(labels)

def run_simple_classifiers(train_loader, test_loader, class_names, device):
    """
    Runs a suite of simple classifiers on extracted features after finding
    best hyperparameters using GridSearchCV.
    """
    print("\n--- Running Simple Classifiers with Hyperparameter Tuning ---")
    
    print("1. Extracting features with MobileNetV2...")
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
    feature_extractor = nn.Sequential(backbone, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()).to(device)

    X_train, y_train = extract_features(train_loader, feature_extractor, device)
    X_test, y_test = extract_features(test_loader, feature_extractor, device)

    print("\n2. Defining models and parameter grids for GridSearch...")

    param_grids = {
        "kNN": {
            "model": KNeighborsClassifier(n_jobs=-1),
            "params": {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski', 'cosine']
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 10, 20],
                'criterion': ['gini', 'entropy']
            }
        },
        "MLPClassifier": {
            "model": MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
            "params": {
                'hidden_layer_sizes': [(512,), (1024,), (512, 256)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.0001]
            }
        }
    }

    for name, config in param_grids.items():
        print(f"\n{'='*20} Running GridSearchCV for {name} {'='*20}")
        grid_search = GridSearchCV(config["model"], config["params"], cv=3, verbose=2, scoring='accuracy', n_jobs=-1)
        
        grid_search.fit(X_train, y_train)

        print(f"\n--- Best parameters found for {name}: ---")
        print(grid_search.best_params_)
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

        print(f"\n--- Final evaluation for best {name} model on test set ---")
        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test)

        overall_accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall Accuracy for best {name}: {overall_accuracy:.4f} ({overall_accuracy:.2%})")

        plot_confusion_matrix_and_report(y_test, y_pred, class_names, f"best_{name}")

    print("\n--- Simple Classifiers Finished ---")